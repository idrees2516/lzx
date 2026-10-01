/* lzx zkVM guest memory map (8 KiB flat image, 1024 u64 words):
 *   0x0000-0x0BFF  .data/.bss   (orderbook, positions, trade buffer, I/O buffers)
 *   0x0C00-0x0FFF  stack        (entry sets sp = 0x0FC0, grows down, < 1 KiB)
 *   0x1000-0x1BFF  .text/.rodata (program, <= 3 KiB, ENTRY(_start) at 0x1000)
 *   0x1C00-0x1DFF  INPUT region  (512 B, 64 u64 words, read-only)
 *   0x1E00-0x1FFF  OUTPUT region (512 B, 64 u64 words, guest-written)
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
  DATA  (rw) : ORIGIN = 0x0000, LENGTH = 0x0C00
  STACK (rw) : ORIGIN = 0x0C00, LENGTH = 0x0400
  TEXT  (rx) : ORIGIN = 0x1000, LENGTH = 0x0C00
  INREG (r)  : ORIGIN = 0x1C00, LENGTH = 0x0200
  OUTREG(rw) : ORIGIN = 0x1E00, LENGTH = 0x0200
}

SECTIONS
{
  .text : ALIGN(4) {
    KEEP(*(.text._start))
    *(.text .text.*)
    *(.rodata .rodata.* .srodata .srodata.*)
  } > TEXT

  .data : ALIGN(8) {
    /* gp anchor for small data (set from the entry stub before any use) */
    __global_pointer$ = . + 0x800;
    *(.data .data.* .sdata .sdata.*)
  } > DATA

  .bss (NOLOAD) : ALIGN(8) {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.*)
    . = ALIGN(8);
    __bss_end = .;
  } > DATA

  /DISCARD/ : {
    *(.eh_frame .eh_frame_hdr)
    *(.note .note.*)
    *(.comment)
    *(.riscv.attributes)
  }
}
