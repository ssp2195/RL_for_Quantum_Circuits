OPENQASM 3.0;
include "stdgates.inc";
qubit[9] q;
bit[4] c;
// prepare
h q[8];
t q[8];
cx q[4], q[8];
tdg q[8];
cx q[0], q[8];
t q[8];
cx q[4], q[8];
tdg q[8];
h q[8];
sdg q[8];
h q[5];
t q[5];
cx q[8], q[5];
tdg q[5];
cx q[1], q[5];
t q[5];
cx q[8], q[5];
tdg q[5];
h q[5];
sdg q[5];
h q[6];
t q[6];
cx q[8], q[6];
tdg q[6];
cx q[2], q[6];
t q[6];
cx q[8], q[6];
tdg q[6];
h q[6];
sdg q[6];
h q[7];
t q[7];
cx q[8], q[7];
tdg q[7];
cx q[3], q[7];
t q[7];
cx q[8], q[7];
tdg q[7];
h q[7];
sdg q[7];
barrier q;
// consumer
s q[5];
s q[5];
h q[6];
cx q[5], q[6];
h q[6];
h q[7];
cx q[5], q[7];
h q[7];
h q[7];
cx q[6], q[7];
h q[7];
barrier q;
// measure_products
h q[5];
c[0] = measure q[5];
if (c[0] == 1) {
  h q[5];
  s q[5];
  s q[5];
  h q[5];
}
h q[6];
c[1] = measure q[6];
if (c[1] == 1) {
  h q[6];
  s q[6];
  s q[6];
  h q[6];
}
h q[7];
c[2] = measure q[7];
if (c[2] == 1) {
  h q[7];
  s q[7];
  s q[7];
  h q[7];
}
barrier q;
// correct_products
if (c[0] == 1) {
  h q[8];
  cx q[1], q[8];
  h q[8];
}
if (c[1] == 1) {
  h q[8];
  cx q[2], q[8];
  h q[8];
}
if (c[2] == 1) {
  h q[8];
  cx q[3], q[8];
  h q[8];
}
barrier q;
// measure_helpers
h q[8];
c[3] = measure q[8];
if (c[3] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
barrier q;
// correct_helpers
if (c[3] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
