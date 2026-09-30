from autokernel.metrics.ptxas import parse_ptxas, resources_summary

SAMPLE = """ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z10gemm_naivePKfS0_Pfiii' for 'sm_120'
ptxas info    : Function properties for _Z10gemm_naivePKfS0_Pfiii
    16 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads
ptxas info    : Used 37 registers, 4096 bytes smem, used 1 barriers, 384 bytes cmem[0]
ptxas info    : Compile time = 4.332 ms
"""


def test_parse_ptxas_fields():
    res = parse_ptxas(SAMPLE)
    assert len(res["kernels"]) == 1
    k = res["kernels"][0]
    assert k["registers"] == 37
    assert k["smem_bytes"] == 4096
    assert k["spill_stores"] == 8 and k["spill_loads"] == 8 and k["stack_bytes"] == 16
    assert k["barriers"] == 1
    assert k["cmem"]["0"] == 384
    assert k["arch"] == "sm_120"
    assert "gemm_naive" in k["name"]


def test_summary_mentions_registers():
    text = resources_summary(parse_ptxas(SAMPLE))
    assert "37 registers" in text and "4096 B" in text and "16 B spills" in text


def test_empty():
    assert parse_ptxas("")["kernels"] == []
    assert "no kernels" in resources_summary({})
