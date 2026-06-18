//===- HexagonQFPSpillFixup.cpp - qf32 spill legalization ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HVX "qf32" (qfloat) vector values carry architectural extended state that is
// not preserved by a plain vmem spill/reload: storing a qf32 register saves
// only its packed 32-bit-per-lane payload, and reloading it (a write without a
// .qf32 qualifier) resets the extended state. A value spilled while qf32 and
// later consumed as .qf32 is therefore silently corrupted -- observed as huge
// garbage in f32 matmul reductions under register pressure.
//
// This pass runs after register allocation, while spills are still
// PS_vstorerv_ai / PS_vloadrv_ai pseudos carrying frame indices (before PEI and
// before post-RA pseudo expansion).  It makes such spills value-preserving
// WITHOUT requiring any extra register:
//
//   * At a spill store of a qf32 value, convert it to IEEE sf in place
//     (V6_vconv_sf_qf32 with Vd == Vu) just before the store, so the slot holds
//     a self-contained sf value that round-trips through memory.
//   * At the matching reload, retype the qf32-reading consumers so they read the
//     reloaded value as .sf (V6_vadd_qf32 -> V6_vadd_qf32_mix, V6_vadd_qf32_mix
//     -> V6_vadd_sf).  This needs no reload-side conversion and no v81-only
//     opcode, and introduces no new register operand.
//
// (B1) A small forward dataflow classifies every HVX vector register's value as
// QF32 (defined by a qf32 producer, or copied from one), NotQF32 (anything else
// -- sf / int / permuted bits, which read back correctly), or MaybeQF32 (a
// value reloaded from memory, a vmux, or a join conflict -- could be an
// unconverted qf32). The dataflow is cross-block.
//
// (A1) Spill slots are reused by stack-slot-coloring for several disjoint live
// ranges, so one slot may carry both a qf32 accumulator and unrelated sf/int
// values. The pass reasons per value-flow rather than per whole slot: it
// converts only the QF32 stores, leaves NotQF32 stores untouched, and at the
// reloads retypes only the operands read as .qf32 (leaving .sf reads alone). By
// type-consistency a .qf32-read reload only reads a qf32 store's value (which
// was converted), so this is sound.
//
// A slot is transformed only if it has at least one (killed) QF32 store, no
// MaybeQF32 store, and every reload's consumers are retypeable within the block;
// otherwise it is left untouched. The pass never miscompiles -- it only fixes
// what it can prove. (Cases it cannot prove -- chained spills, the reloaded
// value flowing through a copy, or values live-out of the block -- are left for
// a more global / type-modeling approach.)
//
// Note that this pass does not achieve IEEE-754 compliance as described in:
// https://docs.qualcomm.com/doc/80-N2040-61/topic/hvx-floating-point.html#handling-the-extended-state-of-hvx-floating-point
// Full compliance would require normalization after the reload, which needs an
// additional register and is therefore not suitable to be amended
// post-register-allocation. This pass maintains correctness at the price of
// some precision loss (qf32 extended range collapsed to sf at spill points).
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "hexagon-qfp-spill-fixup"

static cl::opt<bool> DisableQFPSpillFixup(
    "disable-hexagon-qfp-spill-fixup", cl::Hidden, cl::init(false),
    cl::desc("Disable the Hexagon qf32 spill legalization pass."));

namespace {

// Per-register value classification tracked by the forward dataflow.
//   Top      : not yet seen on this path (dataflow identity element).
//   QF32     : holds a qf32 value (spill must convert it).
//   NotQF32  : holds a non-qf32 value (sf / int / permuted bits) -- reads back
//              correctly as-is; safe to spill/reload without conversion.
//   MaybeQF32: could be an unconverted qf32 (reloaded from memory, a vmux, or a
//              control-flow join of QF32 and NotQF32) -- treated conservatively.
enum class QFKind : uint8_t { Top, QF32, NotQF32, MaybeQF32 };

static QFKind meetKind(QFKind A, QFKind B) {
  if (A == QFKind::Top)
    return B;
  if (B == QFKind::Top)
    return A;
  if (A == B)
    return A;
  return QFKind::MaybeQF32; // QF32 vs NotQF32, or anything with MaybeQF32
}

// HVX float ops whose single-vector (HvxVR) destination is .qf32. Must be
// complete: a missed producer would be (mis)classified NotQF32 and left
// unconverted. V6_vconv_sf_qf32 is excluded (it produces IEEE sf).
static bool producesQf32(unsigned Opc) {
  switch (Opc) {
  case Hexagon::V6_vadd_sf:
  case Hexagon::V6_vadd_qf32:
  case Hexagon::V6_vadd_qf32_mix:
  case Hexagon::V6_vsub_sf:
  case Hexagon::V6_vsub_qf32:
  case Hexagon::V6_vsub_qf32_mix:
  case Hexagon::V6_vmpy_qf32:
  case Hexagon::V6_vmpy_qf32_sf:
  case Hexagon::V6_vabs_qf32_sf:
  case Hexagon::V6_vabs_qf32_qf32:
  case Hexagon::V6_vneg_qf32_sf:
  case Hexagon::V6_vneg_qf32_qf32:
  case Hexagon::V6_vilog2_qf32:
  case Hexagon::V6_vconv_qf32_sf:
    return true;
  default:
    return false;
  }
}

// Single-vector vmem reload pseudos/opcodes.
static bool isVectorReload(unsigned Opc) {
  switch (Opc) {
  case Hexagon::PS_vloadrv_ai:
  case Hexagon::V6_vL32b_ai:
  case Hexagon::V6_vL32Ub_ai:
  case Hexagon::V6_vL32b_nt_ai:
    return true;
  default:
    return false;
  }
}

// Is operand index `Idx` of `Opc` read as .qf32 (vs .sf / raw)?
static bool readsOperandAsQf32(unsigned Opc, unsigned Idx) {
  switch (Opc) {
  case Hexagon::V6_vadd_qf32:
  case Hexagon::V6_vsub_qf32:
  case Hexagon::V6_vmpy_qf32:
    return Idx == 1 || Idx == 2;
  case Hexagon::V6_vadd_qf32_mix: // (qf32, sf)
  case Hexagon::V6_vsub_qf32_mix:
    return Idx == 1;
  case Hexagon::V6_vconv_sf_qf32: // (qf32) -> sf
  case Hexagon::V6_vabs_qf32_qf32:
  case Hexagon::V6_vneg_qf32_qf32:
  case Hexagon::V6_vilog2_qf32:
    return Idx == 1;
  default:
    return false;
  }
}

using QFStateMap = DenseMap<Register, QFKind>;

static bool sameState(const QFStateMap &A, const QFStateMap &B) {
  if (A.size() != B.size())
    return false;
  for (const auto &KV : A) {
    auto It = B.find(KV.first);
    if (It == B.end() || It->second != KV.second)
      return false;
  }
  return true;
}

struct HexagonQFPSpillFixup : public MachineFunctionPass {
  static char ID;
  HexagonQFPSpillFixup() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "Hexagon qf32 spill legalization";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  // How a use of the reloaded value should be handled.
  enum class UseKind { Retype, Leave, Bail };

  const HexagonInstrInfo *HII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

  bool isHvxVR(Register R) const {
    return R.isPhysical() && Hexagon::HvxVRRegClass.contains(R);
  }

  static QFKind lookup(const QFStateMap &S, Register R) {
    auto It = S.find(R);
    return It == S.end() ? QFKind::Top : It->second;
  }

  QFKind defKind(const MachineInstr &MI) const;
  void transfer(const MachineInstr &MI, QFStateMap &S) const;

  UseKind classifyUse(const MachineInstr &MI, Register Reg) const;
  void retypeConsumer(MachineInstr &MI, Register Reg) const;
};

char HexagonQFPSpillFixup::ID = 0;

} // namespace

INITIALIZE_PASS(HexagonQFPSpillFixup, DEBUG_TYPE,
                "Hexagon qf32 spill legalization", false, false)

QFKind HexagonQFPSpillFixup::defKind(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  if (producesQf32(Opc))
    return QFKind::QF32;
  if (isVectorReload(Opc) || Opc == Hexagon::V6_vmux)
    return QFKind::MaybeQF32; // could carry an unconverted qf32
  return QFKind::NotQF32;      // sf / int / permuted bits -- read back as-is
}

void HexagonQFPSpillFixup::transfer(const MachineInstr &MI,
                                    QFStateMap &S) const {
  // A call clobbers caller-saved vector registers.
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isRegMask())
      continue;
    for (MCPhysReg R : Hexagon::HvxVRRegClass)
      if (MO.clobbersPhysReg(R))
        S[R] = QFKind::MaybeQF32;
  }

  if (MI.isCopy()) {
    const MachineOperand &Dst = MI.getOperand(0);
    const MachineOperand &Src = MI.getOperand(1);
    if (Dst.isReg() && isHvxVR(Dst.getReg()))
      S[Dst.getReg()] = (Src.isReg() && isHvxVR(Src.getReg()))
                            ? lookup(S, Src.getReg())
                            : QFKind::MaybeQF32;
    return;
  }

  QFKind K = defKind(MI);
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef() && isHvxVR(MO.getReg()))
      S[MO.getReg()] = K;
}

HexagonQFPSpillFixup::UseKind
HexagonQFPSpillFixup::classifyUse(const MachineInstr &MI, Register Reg) const {
  if (MI.isCopy())
    return UseKind::Bail; // can't follow the reloaded value through a copy
  unsigned Opc = MI.getOpcode();
  for (unsigned I = 1, E = MI.getNumOperands(); I < E; ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isUse() || MO.getReg() != Reg)
      continue;
    if (readsOperandAsQf32(Opc, I)) {
      // Only the qf32 adds can be demoted to read sf.
      if (Opc == Hexagon::V6_vadd_qf32 || Opc == Hexagon::V6_vadd_qf32_mix)
        return UseKind::Retype;
      return UseKind::Bail; // qf32-read we don't know how to retype
    }
    // operand read as .sf / raw -> fine, keep checking the others
  }
  return UseKind::Leave;
}

void HexagonQFPSpillFixup::retypeConsumer(MachineInstr &MI, Register Reg) const {
  unsigned Opc = MI.getOpcode();
  bool InOp1 = MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == Reg;
  bool InOp2 = MI.getOperand(2).isReg() && MI.getOperand(2).getReg() == Reg;

  if (Opc == Hexagon::V6_vadd_qf32) {
    if (InOp1 && InOp2) {
      MI.setDesc(HII->get(Hexagon::V6_vadd_sf));
    } else if (InOp2) {
      MI.setDesc(HII->get(Hexagon::V6_vadd_qf32_mix));
    } else { // reload is op1; commute into the sf slot (op2) of the mix form
      MachineOperand &O1 = MI.getOperand(1);
      MachineOperand &O2 = MI.getOperand(2);
      Register R1 = O1.getReg();
      bool K1 = O1.isKill();
      Register R2 = O2.getReg();
      bool K2 = O2.isKill();
      O1.setReg(R2);
      O1.setIsKill(K2);
      O2.setReg(R1);
      O2.setIsKill(K1);
      MI.setDesc(HII->get(Hexagon::V6_vadd_qf32_mix));
    }
    return;
  }
  if (Opc == Hexagon::V6_vadd_qf32_mix && InOp1)
    MI.setDesc(HII->get(Hexagon::V6_vadd_sf));
}

bool HexagonQFPSpillFixup::runOnMachineFunction(MachineFunction &MF) {
  if (DisableQFPSpillFixup || skipFunction(MF.getFunction()))
    return false;

  const HexagonSubtarget &ST = MF.getSubtarget<HexagonSubtarget>();
  if (!ST.useHVXOps() || !ST.useHVXQFloatOps())
    return false;

  HII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();

  // ---- B1: forward dataflow for the per-register classification. ----
  DenseMap<const MachineBasicBlock *, QFStateMap> In, Out;
  bool Changed = true;
  unsigned Guard = 0;
  while (Changed && Guard++ < 50) {
    Changed = false;
    for (MachineBasicBlock &MBB : MF) {
      QFStateMap NewIn;
      bool First = true;
      for (MachineBasicBlock *Pred : MBB.predecessors()) {
        auto It = Out.find(Pred);
        if (It == Out.end())
          continue;
        if (First) {
          NewIn = It->second;
          First = false;
        } else {
          QFStateMap Merged;
          for (auto &KV : It->second)
            Merged[KV.first] = meetKind(lookup(NewIn, KV.first), KV.second);
          for (auto &KV : NewIn)
            if (!It->second.count(KV.first))
              Merged[KV.first] = KV.second; // meet(x, Top) = x
          NewIn = std::move(Merged);
        }
      }
      QFStateMap NewOut = NewIn;
      for (const MachineInstr &MI : MBB)
        transfer(MI, NewOut);
      if (!sameState(In[&MBB], NewIn)) {
        In[&MBB] = std::move(NewIn);
        Changed = true;
      }
      if (!sameState(Out[&MBB], NewOut)) {
        Out[&MBB] = std::move(NewOut);
        Changed = true;
      }
    }
  }

  // ---- Gather per-frame-index stores/reloads. ----
  struct SlotInfo {
    SmallVector<MachineInstr *, 4> Qf32Stores; // killed qf32 stores to convert
    SmallVector<MachineInstr *, 4> Reloads;
    bool HasBadStore = false; // a MaybeQF32 / un-killed-QF32 store
    bool ReloadsOk = true;    // every reload's qf32-reads are retypeable
  };
  DenseMap<int, SlotInfo> Slots;

  for (MachineBasicBlock &MBB : MF) {
    QFStateMap S = In[&MBB];
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();

      if (Opc == Hexagon::PS_vstorerv_ai && MI.getOperand(0).isFI()) {
        const MachineOperand &Src = MI.getOperand(2);
        QFKind K = lookup(S, Src.getReg());
        SlotInfo &SI = Slots[MI.getOperand(0).getIndex()];
        if (K == QFKind::QF32) {
          if (Src.isKill())
            SI.Qf32Stores.push_back(&MI);
          else
            SI.HasBadStore = true;
        } else if (K == QFKind::MaybeQF32 || K == QFKind::Top) {
          SI.HasBadStore = true; // might be an unconverted qf32
        }
        // NotQF32: leave it.
      } else if (Opc == Hexagon::PS_vloadrv_ai && MI.getOperand(1).isFI()) {
        SlotInfo &SI = Slots[MI.getOperand(1).getIndex()];
        SI.Reloads.push_back(&MI);
        Register D = MI.getOperand(0).getReg();
        bool Bounded = false;
        for (MachineInstr *U = MI.getNextNode(); U; U = U->getNextNode()) {
          if (U->readsRegister(D, TRI) &&
              classifyUse(*U, D) == UseKind::Bail) {
            SI.ReloadsOk = false;
            Bounded = true;
            break;
          }
          if (U->killsRegister(D, TRI) || U->definesRegister(D, TRI)) {
            Bounded = true;
            break;
          }
        }
        if (!Bounded)
          SI.ReloadsOk = false; // reloaded value escapes the block
      }

      transfer(MI, S);
    }
  }

  // ---- Transform fully-provable slots. ----
  bool Modified = false;
  for (auto &KV : Slots) {
    SlotInfo &SI = KV.second;
    if (SI.HasBadStore || SI.Qf32Stores.empty() || !SI.ReloadsOk ||
        SI.Reloads.empty())
      continue;

    for (MachineInstr *Store : SI.Qf32Stores) {
      Register R = Store->getOperand(2).getReg();
      BuildMI(*Store->getParent(), *Store, Store->getDebugLoc(),
              HII->get(Hexagon::V6_vconv_sf_qf32), R)
          .addReg(R, RegState::Kill);
      Modified = true;
    }

    for (MachineInstr *Reload : SI.Reloads) {
      Register D = Reload->getOperand(0).getReg();
      for (MachineInstr *U = Reload->getNextNode(); U; U = U->getNextNode()) {
        if (U->readsRegister(D, TRI) && classifyUse(*U, D) == UseKind::Retype)
          retypeConsumer(*U, D);
        if (U->killsRegister(D, TRI) || U->definesRegister(D, TRI))
          break;
      }
    }
  }

  return Modified;
}

FunctionPass *llvm::createHexagonQFPSpillFixup() {
  return new HexagonQFPSpillFixup();
}
