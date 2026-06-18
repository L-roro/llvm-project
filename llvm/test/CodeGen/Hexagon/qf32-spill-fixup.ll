; Verify that HexagonQFPSpillFixup makes a spill of a live qf32 value
; value-preserving: the qf32 value is converted to IEEE sf *before* the spill
; store, and its reload is consumed as .sf -- so the qf32 extended state that a
; plain vmem spill/reload would drop is never relied upon.
;
; RUN: llc -mtriple=hexagon-unknown-unknown-elf -mcpu=hexagonv79 \
; RUN:   -mattr=+hvxv79,+hvx-length128b --relocation-model=pic < %s -o - \
; RUN:   | FileCheck %s --check-prefix=FIXED
;
; With the pass disabled, the buggy pattern is emitted: the qf32
; value is spilled with a plain vector store and reloaded/consumed as .qf32.
; In case someone fixes this in the future in a more principled way, failure
; of this test will indicate that the pass can be removed.
;
; RUN: llc -mtriple=hexagon-unknown-unknown-elf -mcpu=hexagonv79 \
; RUN:   -mattr=+hvxv79,+hvx-length128b --relocation-model=pic \
; RUN:   -disable-hexagon-qfp-spill-fixup < %s -o - \
; RUN:   | FileCheck %s --check-prefix=BUG

target triple = "hexagon-unknown-unknown-elf"

declare void @opaque_call()

; A qf32 multiply result is kept live across an opaque call, forcing a spill,
; then consumed by an add.

; FIXED-LABEL: qf32_call_spill:
; FIXED:      [[Q:v[0-9]+]].qf32 = vmpy
; FIXED:      [[Q]].sf = [[Q]].qf32
; FIXED:      vmem(r29+#{{[0-9]+}}) = [[Q]]{{(\.new)?}}
; FIXED:      = vadd({{v[0-9]+}}.sf,{{v[0-9]+}}.sf)

; BUG-LABEL: qf32_call_spill:
; BUG:      [[Q:v[0-9]+]].qf32 = vmpy
; BUG:      vmem(r29+#{{[0-9]+}}) = [[Q]]
; BUG:      = vadd({{v[0-9]+}}.qf32,{{v[0-9]+}}.sf)

define void @qf32_call_spill(ptr nocapture %out, <32 x float> %a, <32 x float> %b, <32 x float> %c) #0 {
entry:
  %q = fmul <32 x float> %a, %b
  call void @opaque_call()
  %r = fadd <32 x float> %q, %c
  store <32 x float> %r, ptr %out, align 128
  ret void
}

; Two qf32 products are both kept live across the call, so both are spilled, then
; summed.  The consumer is the *pure* V6_vadd_qf32 (BOTH operands read as .qf32) --
; a different shape from the single-spill case above (which lowers to
; V6_vadd_qf32_mix).  The pass converts both spilled values to sf and demotes the
; add all the way to V6_vadd_sf, so it reads both reloaded operands as .sf.

; FIXED-LABEL: two_qf32:
; FIXED:      = vadd({{v[0-9]+}}.sf,{{v[0-9]+}}.sf)

; BUG-LABEL: two_qf32:
; BUG:      = vadd({{v[0-9]+}}.qf32,{{v[0-9]+}}.qf32)

define void @two_qf32(ptr nocapture %out, <32 x float> %a, <32 x float> %b, <32 x float> %c, <32 x float> %d) #0 {
entry:
  %q1 = fmul <32 x float> %a, %b
  %q2 = fmul <32 x float> %c, %d
  call void @opaque_call()
  %r = fadd <32 x float> %q1, %q2
  store <32 x float> %r, ptr %out, align 128
  ret void
}

; The classification is cross-block: the qf32 value is produced in the entry
; block, spilled while live across the calls on both arms, and reloaded/consumed
; only in the join block.  The pass still converts the spill (in entry) to sf and
; retypes the consumer (in join) to read .sf, exercising the multi-block dataflow.

; FIXED-LABEL: qf32_cross_block:
; FIXED:      [[Q:v[0-9]+]].qf32 = vmpy
; FIXED:      {{v[0-9]+}}.sf = [[Q]].qf32
; FIXED:      = vadd({{v[0-9]+}}.sf,{{v[0-9]+}}.sf)

; BUG-LABEL: qf32_cross_block:
; BUG:      = vadd({{v[0-9]+}}.qf32,{{v[0-9]+}}.sf)

define void @qf32_cross_block(ptr nocapture %out, <32 x float> %a, <32 x float> %b, <32 x float> %c, i32 %n) #0 {
entry:
  %q = fmul <32 x float> %a, %b
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %then, label %else
then:
  call void @opaque_call()
  br label %join
else:
  call void @opaque_call()
  br label %join
join:
  %r = fadd <32 x float> %q, %c
  store <32 x float> %r, ptr %out, align 128
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv79" "target-features"="+hvxv79,+hvx-length128b" }
