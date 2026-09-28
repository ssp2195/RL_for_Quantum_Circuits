OPENQASM 3.0;
include "stdgates.inc";
qubit[11] q;
bit[4] c;
// prepare
cx q[4], q[1];
h q[8];
t q[8];
cx q[1], q[8];
tdg q[8];
cx q[0], q[8];
t q[8];
cx q[1], q[8];
tdg q[8];
h q[8];
sdg q[8];
barrier q;
// consumer
h q[7];
cx q[8], q[7];
h q[7];
barrier q;
// measure_factor_0
h q[8];
c[0] = measure q[8];
if (c[0] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
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
cx q[4], q[1];
barrier q;
// prepare
h q[9];
t q[9];
cx q[4], q[9];
tdg q[9];
cx q[1], q[9];
t q[9];
cx q[4], q[9];
tdg q[9];
h q[9];
sdg q[9];
h q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
cx q[2], q[10];
t q[10];
cx q[4], q[10];
tdg q[10];
h q[10];
sdg q[10];
cx q[9], q[3];
cx q[10], q[3];
h q[8];
t q[8];
cx q[3], q[8];
tdg q[8];
cx q[0], q[8];
t q[8];
cx q[3], q[8];
tdg q[8];
h q[8];
sdg q[8];
barrier q;
// consumer
h q[6];
cx q[8], q[6];
h q[6];
barrier q;
// measure_factor_1
h q[8];
c[1] = measure q[8];
if (c[1] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
barrier q;
// correct_factor_1
if (c[1] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
barrier q;
// restore_factor_1
cx q[10], q[3];
cx q[9], q[3];
barrier q;
// measure_features_0
h q[9];
c[2] = measure q[9];
if (c[2] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
}
h q[10];
c[3] = measure q[10];
if (c[3] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_features_0
if (c[2] == 1) {
  h q[4];
  cx q[1], q[4];
  h q[4];
}
if (c[3] == 1) {
  h q[4];
  cx q[2], q[4];
  h q[4];
}
