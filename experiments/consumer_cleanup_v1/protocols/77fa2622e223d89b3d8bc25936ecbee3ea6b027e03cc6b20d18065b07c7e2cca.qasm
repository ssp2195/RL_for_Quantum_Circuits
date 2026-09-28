OPENQASM 3.0;
include "stdgates.inc";
qubit[7] q;
bit[2] c;
// prepare
cx q[3], q[1];
cx q[3], q[2];
h q[6];
t q[6];
cx q[2], q[6];
tdg q[6];
cx q[1], q[6];
t q[6];
cx q[2], q[6];
tdg q[6];
h q[6];
sdg q[6];
cx q[3], q[2];
cx q[3], q[1];
cx q[3], q[1];
cx q[6], q[1];
h q[5];
t q[5];
cx q[1], q[5];
tdg q[5];
cx q[0], q[5];
t q[5];
cx q[1], q[5];
tdg q[5];
h q[5];
sdg q[5];
barrier q;
// consumer
h q[4];
cx q[5], q[4];
h q[4];
barrier q;
// measure_factor_0
h q[5];
c[0] = measure q[5];
if (c[0] == 1) {
  h q[5];
  s q[5];
  s q[5];
  h q[5];
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
cx q[6], q[1];
cx q[3], q[1];
barrier q;
// measure_features_0
h q[6];
c[1] = measure q[6];
if (c[1] == 1) {
  h q[6];
  s q[6];
  s q[6];
  h q[6];
}
barrier q;
// correct_features_0
cx q[3], q[1];
cx q[3], q[2];
if (c[1] == 1) {
  h q[2];
  cx q[1], q[2];
  h q[2];
}
cx q[3], q[2];
cx q[3], q[1];
