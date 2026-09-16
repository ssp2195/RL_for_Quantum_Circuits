OPENQASM 3.0;
include "stdgates.inc";
qubit[17] q;
bit[10] c;
// prepare
h q[15];
t q[15];
cx q[5], q[15];
tdg q[15];
cx q[0], q[15];
t q[15];
cx q[5], q[15];
tdg q[15];
h q[15];
sdg q[15];
h q[16];
t q[16];
cx q[6], q[16];
tdg q[16];
cx q[0], q[16];
t q[16];
cx q[6], q[16];
tdg q[16];
h q[16];
sdg q[16];
h q[7];
t q[7];
cx q[15], q[7];
tdg q[7];
cx q[1], q[7];
t q[7];
cx q[15], q[7];
tdg q[7];
h q[7];
sdg q[7];
h q[10];
t q[10];
cx q[16], q[10];
tdg q[10];
cx q[2], q[10];
t q[10];
cx q[16], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[9];
t q[9];
cx q[15], q[9];
tdg q[9];
cx q[2], q[9];
t q[9];
cx q[15], q[9];
tdg q[9];
h q[9];
sdg q[9];
h q[8];
t q[8];
cx q[16], q[8];
tdg q[8];
cx q[1], q[8];
t q[8];
cx q[16], q[8];
tdg q[8];
h q[8];
sdg q[8];
h q[11];
t q[11];
cx q[15], q[11];
tdg q[11];
cx q[3], q[11];
t q[11];
cx q[15], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[14];
t q[14];
cx q[16], q[14];
tdg q[14];
cx q[4], q[14];
t q[14];
cx q[16], q[14];
tdg q[14];
h q[14];
sdg q[14];
h q[13];
t q[13];
cx q[15], q[13];
tdg q[13];
cx q[4], q[13];
t q[13];
cx q[15], q[13];
tdg q[13];
h q[13];
sdg q[13];
h q[12];
t q[12];
cx q[16], q[12];
tdg q[12];
cx q[3], q[12];
t q[12];
cx q[16], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
s q[9];
s q[9];
s q[12];
s q[12];
h q[6];
cx q[0], q[6];
h q[6];
h q[14];
cx q[2], q[14];
h q[14];
h q[10];
cx q[4], q[10];
h q[10];
h q[12];
cx q[7], q[12];
h q[12];
h q[11];
cx q[8], q[11];
h q[11];
h q[14];
cx q[9], q[14];
h q[14];
h q[13];
cx q[10], q[13];
h q[13];
barrier q;
// measure_products
h q[7];
c[0] = measure q[7];
if (c[0] == 1) {
  h q[7];
  s q[7];
  s q[7];
  h q[7];
}
h q[8];
c[1] = measure q[8];
if (c[1] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
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
barrier q;
// correct_products
if (c[0] == 1) {
  h q[15];
  cx q[1], q[15];
  h q[15];
}
if (c[1] == 1) {
  h q[16];
  cx q[1], q[16];
  h q[16];
}
if (c[2] == 1) {
  h q[15];
  cx q[2], q[15];
  h q[15];
}
if (c[3] == 1) {
  h q[16];
  cx q[2], q[16];
  h q[16];
}
if (c[4] == 1) {
  h q[15];
  cx q[3], q[15];
  h q[15];
}
if (c[5] == 1) {
  h q[16];
  cx q[3], q[16];
  h q[16];
}
if (c[6] == 1) {
  h q[15];
  cx q[4], q[15];
  h q[15];
}
if (c[7] == 1) {
  h q[16];
  cx q[4], q[16];
  h q[16];
}
barrier q;
// measure_helpers
h q[15];
c[8] = measure q[15];
if (c[8] == 1) {
  h q[15];
  s q[15];
  s q[15];
  h q[15];
}
h q[16];
c[9] = measure q[16];
if (c[9] == 1) {
  h q[16];
  s q[16];
  s q[16];
  h q[16];
}
barrier q;
// correct_helpers
if (c[8] == 1) {
  h q[5];
  cx q[0], q[5];
  h q[5];
}
if (c[9] == 1) {
  h q[6];
  cx q[0], q[6];
  h q[6];
}
