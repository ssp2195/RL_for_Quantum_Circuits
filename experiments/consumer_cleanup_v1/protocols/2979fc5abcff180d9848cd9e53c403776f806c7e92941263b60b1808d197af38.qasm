OPENQASM 3.0;
include "stdgates.inc";
qubit[13] q;
bit[70] c;
// consumer
h q[7];
cx q[0], q[7];
h q[7];
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
barrier q;
// consumer
h q[7];
cx q[10], q[7];
h q[7];
barrier q;
// measure_reference_0
h q[10];
c[0] = measure q[10];
if (c[0] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_0
if (c[0] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
barrier q;
// consumer
h q[7];
cx q[10], q[7];
h q[7];
barrier q;
// measure_reference_1
h q[10];
c[1] = measure q[10];
if (c[1] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_1
if (c[1] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
h q[10];
sdg q[10];
barrier q;
// consumer
h q[7];
cx q[10], q[7];
h q[7];
barrier q;
// measure_reference_2
h q[10];
c[2] = measure q[10];
if (c[2] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_2
if (c[2] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
h q[10];
sdg q[10];
barrier q;
// consumer
h q[7];
cx q[10], q[7];
h q[7];
barrier q;
// measure_reference_3
h q[10];
c[3] = measure q[10];
if (c[3] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_3
if (c[3] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[5], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[5], q[10];
tdg q[10];
h q[10];
sdg q[10];
barrier q;
// consumer
h q[7];
cx q[10], q[7];
h q[7];
barrier q;
// measure_reference_4
h q[10];
c[4] = measure q[10];
if (c[4] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_4
if (c[4] == 1) {
  h q[5];
  cx q[0], q[5];
  h q[5];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[6], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[6], q[10];
tdg q[10];
h q[10];
sdg q[10];
barrier q;
// consumer
h q[7];
cx q[10], q[7];
h q[7];
barrier q;
// measure_reference_5
h q[10];
c[5] = measure q[10];
if (c[5] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_5
if (c[5] == 1) {
  h q[6];
  cx q[0], q[6];
  h q[6];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
h q[11];
sdg q[11];
barrier q;
// consumer
h q[7];
cx q[11], q[7];
h q[7];
barrier q;
// measure_reference_6
h q[11];
c[6] = measure q[11];
if (c[6] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_6
if (c[6] == 1) {
  h q[6];
  cx q[10], q[6];
  h q[6];
}
barrier q;
// measure_reference_7
h q[10];
c[7] = measure q[10];
if (c[7] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_7
if (c[7] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
h q[11];
sdg q[11];
barrier q;
// consumer
h q[8];
cx q[11], q[8];
h q[8];
barrier q;
// measure_reference_8
h q[11];
c[8] = measure q[11];
if (c[8] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_8
if (c[8] == 1) {
  h q[3];
  cx q[10], q[3];
  h q[3];
}
barrier q;
// measure_reference_9
h q[10];
c[9] = measure q[10];
if (c[9] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_9
if (c[9] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
h q[11];
sdg q[11];
barrier q;
// consumer
h q[8];
cx q[11], q[8];
h q[8];
barrier q;
// measure_reference_10
h q[11];
c[10] = measure q[11];
if (c[10] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_10
if (c[10] == 1) {
  h q[5];
  cx q[10], q[5];
  h q[5];
}
barrier q;
// measure_reference_11
h q[10];
c[11] = measure q[10];
if (c[11] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_11
if (c[11] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[8];
cx q[12], q[8];
h q[8];
barrier q;
// measure_reference_12
h q[12];
c[12] = measure q[12];
if (c[12] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_12
if (c[12] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_13
h q[11];
c[13] = measure q[11];
if (c[13] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_13
if (c[13] == 1) {
  h q[3];
  cx q[10], q[3];
  h q[3];
}
barrier q;
// measure_reference_14
h q[10];
c[14] = measure q[10];
if (c[14] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_14
if (c[14] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[8];
cx q[12], q[8];
h q[8];
barrier q;
// measure_reference_15
h q[12];
c[15] = measure q[12];
if (c[15] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_15
if (c[15] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_16
h q[11];
c[16] = measure q[11];
if (c[16] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_16
if (c[16] == 1) {
  h q[4];
  cx q[10], q[4];
  h q[4];
}
barrier q;
// measure_reference_17
h q[10];
c[17] = measure q[10];
if (c[17] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_17
if (c[17] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[8];
cx q[12], q[8];
h q[8];
barrier q;
// measure_reference_18
h q[12];
c[18] = measure q[12];
if (c[18] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_18
if (c[18] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_19
h q[11];
c[19] = measure q[11];
if (c[19] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_19
if (c[19] == 1) {
  h q[5];
  cx q[10], q[5];
  h q[5];
}
barrier q;
// measure_reference_20
h q[10];
c[20] = measure q[10];
if (c[20] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_20
if (c[20] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[8];
cx q[12], q[8];
h q[8];
barrier q;
// measure_reference_21
h q[12];
c[21] = measure q[12];
if (c[21] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_21
if (c[21] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_22
h q[11];
c[22] = measure q[11];
if (c[22] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_22
if (c[22] == 1) {
  h q[5];
  cx q[10], q[5];
  h q[5];
}
barrier q;
// measure_reference_23
h q[10];
c[23] = measure q[10];
if (c[23] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_23
if (c[23] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[8];
cx q[12], q[8];
h q[8];
barrier q;
// measure_reference_24
h q[12];
c[24] = measure q[12];
if (c[24] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_24
if (c[24] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_25
h q[11];
c[25] = measure q[11];
if (c[25] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_25
if (c[25] == 1) {
  h q[6];
  cx q[10], q[6];
  h q[6];
}
barrier q;
// measure_reference_26
h q[10];
c[26] = measure q[10];
if (c[26] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_26
if (c[26] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[8];
cx q[12], q[8];
h q[8];
barrier q;
// measure_reference_27
h q[12];
c[27] = measure q[12];
if (c[27] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_27
if (c[27] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_28
h q[11];
c[28] = measure q[11];
if (c[28] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_28
if (c[28] == 1) {
  h q[6];
  cx q[10], q[6];
  h q[6];
}
barrier q;
// measure_reference_29
h q[10];
c[29] = measure q[10];
if (c[29] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_29
if (c[29] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
h q[11];
sdg q[11];
barrier q;
// consumer
h q[9];
cx q[11], q[9];
h q[9];
barrier q;
// measure_reference_30
h q[11];
c[30] = measure q[11];
if (c[30] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_30
if (c[30] == 1) {
  h q[4];
  cx q[10], q[4];
  h q[4];
}
barrier q;
// measure_reference_31
h q[10];
c[31] = measure q[10];
if (c[31] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_31
if (c[31] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
h q[11];
sdg q[11];
barrier q;
// consumer
h q[9];
cx q[11], q[9];
h q[9];
barrier q;
// measure_reference_32
h q[11];
c[32] = measure q[11];
if (c[32] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_32
if (c[32] == 1) {
  h q[6];
  cx q[10], q[6];
  h q[6];
}
barrier q;
// measure_reference_33
h q[10];
c[33] = measure q[10];
if (c[33] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_33
if (c[33] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_34
h q[12];
c[34] = measure q[12];
if (c[34] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_34
if (c[34] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_35
h q[11];
c[35] = measure q[11];
if (c[35] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_35
if (c[35] == 1) {
  h q[3];
  cx q[10], q[3];
  h q[3];
}
barrier q;
// measure_reference_36
h q[10];
c[36] = measure q[10];
if (c[36] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_36
if (c[36] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_37
h q[12];
c[37] = measure q[12];
if (c[37] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_37
if (c[37] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_38
h q[11];
c[38] = measure q[11];
if (c[38] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_38
if (c[38] == 1) {
  h q[4];
  cx q[10], q[4];
  h q[4];
}
barrier q;
// measure_reference_39
h q[10];
c[39] = measure q[10];
if (c[39] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_39
if (c[39] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_40
h q[12];
c[40] = measure q[12];
if (c[40] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_40
if (c[40] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_41
h q[11];
c[41] = measure q[11];
if (c[41] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_41
if (c[41] == 1) {
  h q[5];
  cx q[10], q[5];
  h q[5];
}
barrier q;
// measure_reference_42
h q[10];
c[42] = measure q[10];
if (c[42] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_42
if (c[42] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_43
h q[12];
c[43] = measure q[12];
if (c[43] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_43
if (c[43] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_44
h q[11];
c[44] = measure q[11];
if (c[44] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_44
if (c[44] == 1) {
  h q[5];
  cx q[10], q[5];
  h q[5];
}
barrier q;
// measure_reference_45
h q[10];
c[45] = measure q[10];
if (c[45] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_45
if (c[45] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_46
h q[12];
c[46] = measure q[12];
if (c[46] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_46
if (c[46] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_47
h q[11];
c[47] = measure q[11];
if (c[47] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_47
if (c[47] == 1) {
  h q[6];
  cx q[10], q[6];
  h q[6];
}
barrier q;
// measure_reference_48
h q[10];
c[48] = measure q[10];
if (c[48] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_48
if (c[48] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[7], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_49
h q[12];
c[49] = measure q[12];
if (c[49] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_49
if (c[49] == 1) {
  h q[7];
  cx q[11], q[7];
  h q[7];
}
barrier q;
// measure_reference_50
h q[11];
c[50] = measure q[11];
if (c[50] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_50
if (c[50] == 1) {
  h q[6];
  cx q[10], q[6];
  h q[6];
}
barrier q;
// measure_reference_51
h q[10];
c[51] = measure q[10];
if (c[51] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_51
if (c[51] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_52
h q[12];
c[52] = measure q[12];
if (c[52] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_52
if (c[52] == 1) {
  h q[8];
  cx q[11], q[8];
  h q[8];
}
barrier q;
// measure_reference_53
h q[11];
c[53] = measure q[11];
if (c[53] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_53
if (c[53] == 1) {
  h q[3];
  cx q[10], q[3];
  h q[3];
}
barrier q;
// measure_reference_54
h q[10];
c[54] = measure q[10];
if (c[54] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_54
if (c[54] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_55
h q[12];
c[55] = measure q[12];
if (c[55] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_55
if (c[55] == 1) {
  h q[8];
  cx q[11], q[8];
  h q[8];
}
barrier q;
// measure_reference_56
h q[11];
c[56] = measure q[11];
if (c[56] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_56
if (c[56] == 1) {
  h q[4];
  cx q[10], q[4];
  h q[4];
}
barrier q;
// measure_reference_57
h q[10];
c[57] = measure q[10];
if (c[57] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_57
if (c[57] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[1], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_58
h q[12];
c[58] = measure q[12];
if (c[58] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_58
if (c[58] == 1) {
  h q[8];
  cx q[11], q[8];
  h q[8];
}
barrier q;
// measure_reference_59
h q[11];
c[59] = measure q[11];
if (c[59] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_59
if (c[59] == 1) {
  h q[5];
  cx q[10], q[5];
  h q[5];
}
barrier q;
// measure_reference_60
h q[10];
c[60] = measure q[10];
if (c[60] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_60
if (c[60] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[5], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_61
h q[12];
c[61] = measure q[12];
if (c[61] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_61
if (c[61] == 1) {
  h q[8];
  cx q[11], q[8];
  h q[8];
}
barrier q;
// measure_reference_62
h q[11];
c[62] = measure q[11];
if (c[62] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_62
if (c[62] == 1) {
  h q[5];
  cx q[10], q[5];
  h q[5];
}
barrier q;
// measure_reference_63
h q[10];
c[63] = measure q[10];
if (c[63] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_63
if (c[63] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_64
h q[12];
c[64] = measure q[12];
if (c[64] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_64
if (c[64] == 1) {
  h q[8];
  cx q[11], q[8];
  h q[8];
}
barrier q;
// measure_reference_65
h q[11];
c[65] = measure q[11];
if (c[65] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_65
if (c[65] == 1) {
  h q[6];
  cx q[10], q[6];
  h q[6];
}
barrier q;
// measure_reference_66
h q[10];
c[66] = measure q[10];
if (c[66] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_66
if (c[66] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
cx q[10], q[11];
t q[11];
cx q[6], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
cx q[11], q[12];
t q[12];
cx q[8], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_reference_67
h q[12];
c[67] = measure q[12];
if (c[67] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_reference_67
if (c[67] == 1) {
  h q[8];
  cx q[11], q[8];
  h q[8];
}
barrier q;
// measure_reference_68
h q[11];
c[68] = measure q[11];
if (c[68] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_reference_68
if (c[68] == 1) {
  h q[6];
  cx q[10], q[6];
  h q[6];
}
barrier q;
// measure_reference_69
h q[10];
c[69] = measure q[10];
if (c[69] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_reference_69
if (c[69] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
