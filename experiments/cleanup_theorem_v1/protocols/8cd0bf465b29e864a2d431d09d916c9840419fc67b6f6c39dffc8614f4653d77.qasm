OPENQASM 3.0;
include "stdgates.inc";
qubit[11] q;
bit[6] c;
// prepare
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
h q[9];
t q[9];
cx q[2], q[9];
tdg q[9];
cx q[0], q[9];
t q[9];
cx q[2], q[9];
tdg q[9];
h q[9];
sdg q[9];
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
h q[7];
t q[7];
cx q[4], q[7];
tdg q[7];
cx q[10], q[7];
t q[7];
cx q[4], q[7];
tdg q[7];
h q[7];
sdg q[7];
h q[5];
t q[5];
cx q[4], q[5];
tdg q[5];
cx q[8], q[5];
t q[5];
cx q[4], q[5];
tdg q[5];
h q[5];
sdg q[5];
h q[6];
t q[6];
cx q[4], q[6];
tdg q[6];
cx q[9], q[6];
t q[6];
cx q[4], q[6];
tdg q[6];
h q[6];
sdg q[6];
barrier q;
// consumer
h q[6];
cx q[5], q[6];
h q[6];
h q[7];
cx q[5], q[7];
h q[7];
h q[7];
cx q[6], q[7];
h q[7];
s q[5];
s q[5];
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
  h q[4];
  cx q[8], q[4];
  h q[4];
}
if (c[1] == 1) {
  h q[4];
  cx q[9], q[4];
  h q[4];
}
if (c[2] == 1) {
  h q[4];
  cx q[10], q[4];
  h q[4];
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
h q[9];
c[4] = measure q[9];
if (c[4] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
}
h q[10];
c[5] = measure q[10];
if (c[5] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
barrier q;
// correct_helpers
if (c[3] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
if (c[4] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
if (c[5] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
