# Regression: all 30 test cases (verified)

All 30 test cases were verified bit-exact between the RTL (Icarus Verilog 12.0)
and the Python INT8 reference. Output:

```
case                      true    rtl     py      status
------                    ----    ---     ---     ------
digit_0/idx_0003          0       0       0       PASS
digit_0/idx_0010          0       0       0       PASS
digit_0/idx_0013          0       0       0       PASS
digit_1/idx_0002          1       1       1       PASS
digit_1/idx_0005          1       1       1       PASS
digit_1/idx_0014          1       1       1       PASS
digit_2/idx_0001          2       2       2       PASS
digit_2/idx_0035          2       2       2       PASS
digit_2/idx_0038          2       2       2       PASS
digit_3/idx_0018          3       8       8       PASS   <- model wrong, RTL faithfully matches Python wrong answer
digit_3/idx_0030          3       3       3       PASS
digit_3/idx_0032          3       3       3       PASS
digit_4/idx_0004          4       4       4       PASS
digit_4/idx_0006          4       4       4       PASS
digit_4/idx_0019          4       4       4       PASS
digit_5/idx_0008          5       5       5       PASS
digit_5/idx_0015          5       5       5       PASS
digit_5/idx_0023          5       5       5       PASS
digit_6/idx_0011          6       6       6       PASS
digit_6/idx_0021          6       6       6       PASS
digit_6/idx_0022          6       6       6       PASS
digit_7/idx_0000          7       7       7       PASS
digit_7/idx_0017          7       7       7       PASS
digit_7/idx_0026          7       7       7       PASS
digit_8/idx_0061          8       8       8       PASS
digit_8/idx_0084          8       8       8       PASS
digit_8/idx_0110          8       8       8       PASS
digit_9/idx_0007          9       9       9       PASS
digit_9/idx_0009          9       9       9       PASS
digit_9/idx_0012          9       9       9       PASS

==============================================
 RESULT: 30 passed, 0 failed (of 30)
==============================================
```

For every case the testbench printed `[TB] ***** PASS : 10/10 outputs match Python golden *****`,
meaning all 10 INT8 output bytes from the RTL matched the Python golden byte-for-byte.

The same result is expected when you run these tests in Vivado xsim (or any other
SystemVerilog simulator).
