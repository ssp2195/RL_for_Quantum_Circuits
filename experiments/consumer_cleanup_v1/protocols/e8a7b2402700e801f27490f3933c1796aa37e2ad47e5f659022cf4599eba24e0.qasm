OPENQASM 3.0;
include "stdgates.inc";
qubit[16] q;
bit[9] c;
// consumer
h q[7];
cx q[0], q[7];
h q[7];
barrier q;
// prepare
cx q[5], q[3];
h q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
cx q[1], q[11];
t q[11];
cx q[3], q[11];
tdg q[11];
h q[11];
sdg q[11];
cx q[5], q[3];
h q[12];
t q[12];
cx q[5], q[12];
tdg q[12];
cx q[3], q[12];
t q[12];
cx q[5], q[12];
tdg q[12];
h q[12];
sdg q[12];
h q[13];
t q[13];
cx q[6], q[13];
tdg q[13];
cx q[4], q[13];
t q[13];
cx q[6], q[13];
tdg q[13];
h q[13];
sdg q[13];
cx q[6], q[2];
cx q[6], q[4];
h q[14];
t q[14];
cx q[4], q[14];
tdg q[14];
cx q[2], q[14];
t q[14];
cx q[4], q[14];
tdg q[14];
h q[14];
sdg q[14];
cx q[6], q[4];
cx q[6], q[2];
cx q[9], q[7];
cx q[9], q[8];
h q[15];
t q[15];
cx q[8], q[15];
tdg q[15];
cx q[7], q[15];
t q[15];
cx q[8], q[15];
tdg q[15];
h q[15];
sdg q[15];
cx q[9], q[8];
cx q[9], q[7];
cx q[2], q[1];
cx q[3], q[1];
cx q[4], q[1];
cx q[5], q[1];
cx q[6], q[1];
cx q[13], q[1];
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
// measure_factor_0
h q[10];
c[0] = measure q[10];
if (c[0] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_factor_0
if (c[0] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// restore_factor_0
cx q[13], q[1];
cx q[6], q[1];
cx q[5], q[1];
cx q[4], q[1];
cx q[3], q[1];
cx q[2], q[1];
barrier q;
// prepare
cx q[9], q[8];
cx q[15], q[8];
h q[10];
t q[10];
cx q[11], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[11], q[10];
tdg q[10];
h q[10];
sdg q[10];
barrier q;
// consumer
h q[8];
cx q[10], q[8];
h q[8];
barrier q;
// measure_factor_1
h q[10];
c[1] = measure q[10];
if (c[1] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_factor_1
if (c[1] == 1) {
  h q[11];
  cx q[0], q[11];
  h q[11];
}
barrier q;
// restore_factor_1
cx q[15], q[8];
cx q[9], q[8];
barrier q;
// prepare
cx q[14], q[6];
cx q[12], q[6];
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
h q[15];
cx q[10], q[15];
h q[15];
barrier q;
// measure_factor_2
h q[10];
c[2] = measure q[10];
if (c[2] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_factor_2
if (c[2] == 1) {
  h q[6];
  cx q[0], q[6];
  h q[6];
}
barrier q;
// restore_factor_2
cx q[12], q[6];
cx q[14], q[6];
barrier q;
// prepare
cx q[13], q[12];
h q[10];
t q[10];
cx q[12], q[10];
tdg q[10];
cx q[0], q[10];
t q[10];
cx q[12], q[10];
tdg q[10];
h q[10];
sdg q[10];
barrier q;
// consumer
h q[9];
cx q[10], q[9];
h q[9];
barrier q;
// measure_factor_3
h q[10];
c[3] = measure q[10];
if (c[3] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_factor_3
if (c[3] == 1) {
  h q[12];
  cx q[0], q[12];
  h q[12];
}
barrier q;
// restore_factor_3
cx q[13], q[12];
barrier q;
// measure_features_0
h q[11];
c[4] = measure q[11];
if (c[4] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
h q[12];
c[5] = measure q[12];
if (c[5] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
h q[13];
c[6] = measure q[13];
if (c[6] == 1) {
  h q[13];
  s q[13];
  s q[13];
  h q[13];
}
h q[14];
c[7] = measure q[14];
if (c[7] == 1) {
  h q[14];
  s q[14];
  s q[14];
  h q[14];
}
h q[15];
c[8] = measure q[15];
if (c[8] == 1) {
  h q[15];
  s q[15];
  s q[15];
  h q[15];
}
barrier q;
// correct_features_0
cx q[5], q[3];
if (c[4] == 1) {
  h q[3];
  cx q[1], q[3];
  h q[3];
}
cx q[5], q[3];
if (c[5] == 1) {
  h q[5];
  cx q[3], q[5];
  h q[5];
}
if (c[6] == 1) {
  h q[6];
  cx q[4], q[6];
  h q[6];
}
cx q[6], q[2];
cx q[6], q[4];
if (c[7] == 1) {
  h q[4];
  cx q[2], q[4];
  h q[4];
}
cx q[6], q[4];
cx q[6], q[2];
cx q[9], q[7];
cx q[9], q[8];
if (c[8] == 1) {
  h q[8];
  cx q[7], q[8];
  h q[8];
}
cx q[9], q[8];
cx q[9], q[7];
