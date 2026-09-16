OPENQASM 3.0;
include "stdgates.inc";
qubit[20] q;
bit[12] c;
// prepare
h q[18];
t q[18];
cx q[6], q[18];
tdg q[18];
cx q[0], q[18];
t q[18];
cx q[6], q[18];
tdg q[18];
h q[18];
sdg q[18];
h q[19];
t q[19];
cx q[7], q[19];
tdg q[19];
cx q[0], q[19];
t q[19];
cx q[7], q[19];
tdg q[19];
h q[19];
sdg q[19];
h q[8];
t q[8];
cx q[18], q[8];
tdg q[8];
cx q[1], q[8];
t q[8];
cx q[18], q[8];
tdg q[8];
h q[8];
sdg q[8];
h q[11];
t q[11];
cx q[19], q[11];
tdg q[11];
cx q[2], q[11];
t q[11];
cx q[19], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[10];
t q[10];
cx q[18], q[10];
tdg q[10];
cx q[2], q[10];
t q[10];
cx q[18], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[9];
t q[9];
cx q[19], q[9];
tdg q[9];
cx q[1], q[9];
t q[9];
cx q[19], q[9];
tdg q[9];
h q[9];
sdg q[9];
h q[12];
t q[12];
cx q[18], q[12];
tdg q[12];
cx q[3], q[12];
t q[12];
cx q[18], q[12];
tdg q[12];
h q[12];
sdg q[12];
h q[15];
t q[15];
cx q[19], q[15];
tdg q[15];
cx q[4], q[15];
t q[15];
cx q[19], q[15];
tdg q[15];
h q[15];
sdg q[15];
h q[14];
t q[14];
cx q[18], q[14];
tdg q[14];
cx q[4], q[14];
t q[14];
cx q[18], q[14];
tdg q[14];
h q[14];
sdg q[14];
h q[13];
t q[13];
cx q[19], q[13];
tdg q[13];
cx q[3], q[13];
t q[13];
cx q[19], q[13];
tdg q[13];
h q[13];
sdg q[13];
h q[16];
t q[16];
cx q[18], q[16];
tdg q[16];
cx q[5], q[16];
t q[16];
cx q[18], q[16];
tdg q[16];
h q[16];
sdg q[16];
h q[17];
t q[17];
cx q[19], q[17];
tdg q[17];
cx q[5], q[17];
t q[17];
cx q[19], q[17];
tdg q[17];
h q[17];
sdg q[17];
barrier q;
// consumer
s q[9];
s q[9];
s q[12];
s q[12];
s q[15];
s q[15];
h q[14];
cx q[1], q[14];
h q[14];
h q[10];
cx q[4], q[10];
h q[10];
h q[13];
cx q[8], q[13];
h q[13];
h q[15];
cx q[8], q[15];
h q[15];
h q[12];
cx q[9], q[12];
h q[12];
h q[14];
cx q[9], q[14];
h q[14];
h q[15];
cx q[10], q[15];
h q[15];
h q[17];
cx q[10], q[17];
h q[17];
h q[14];
cx q[11], q[14];
h q[14];
h q[16];
cx q[11], q[16];
h q[16];
h q[17];
cx q[12], q[17];
h q[17];
h q[16];
cx q[13], q[16];
h q[16];
barrier q;
// measure_products
h q[8];
c[0] = measure q[8];
if (c[0] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
h q[9];
c[1] = measure q[9];
if (c[1] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
}
h q[10];
c[2] = measure q[10];
if (c[2] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
h q[11];
c[3] = measure q[11];
if (c[3] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
h q[12];
c[4] = measure q[12];
if (c[4] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
h q[13];
c[5] = measure q[13];
if (c[5] == 1) {
  h q[13];
  s q[13];
  s q[13];
  h q[13];
}
h q[14];
c[6] = measure q[14];
if (c[6] == 1) {
  h q[14];
  s q[14];
  s q[14];
  h q[14];
}
h q[15];
c[7] = measure q[15];
if (c[7] == 1) {
  h q[15];
  s q[15];
  s q[15];
  h q[15];
}
h q[16];
c[8] = measure q[16];
if (c[8] == 1) {
  h q[16];
  s q[16];
  s q[16];
  h q[16];
}
h q[17];
c[9] = measure q[17];
if (c[9] == 1) {
  h q[17];
  s q[17];
  s q[17];
  h q[17];
}
barrier q;
// correct_products
if (c[0] == 1) {
  h q[18];
  cx q[1], q[18];
  h q[18];
}
if (c[1] == 1) {
  h q[19];
  cx q[1], q[19];
  h q[19];
}
if (c[2] == 1) {
  h q[18];
  cx q[2], q[18];
  h q[18];
}
if (c[3] == 1) {
  h q[19];
  cx q[2], q[19];
  h q[19];
}
if (c[4] == 1) {
  h q[18];
  cx q[3], q[18];
  h q[18];
}
if (c[5] == 1) {
  h q[19];
  cx q[3], q[19];
  h q[19];
}
if (c[6] == 1) {
  h q[18];
  cx q[4], q[18];
  h q[18];
}
if (c[7] == 1) {
  h q[19];
  cx q[4], q[19];
  h q[19];
}
if (c[8] == 1) {
  h q[18];
  cx q[5], q[18];
  h q[18];
}
if (c[9] == 1) {
  h q[19];
  cx q[5], q[19];
  h q[19];
}
barrier q;
// measure_helpers
h q[18];
c[10] = measure q[18];
if (c[10] == 1) {
  h q[18];
  s q[18];
  s q[18];
  h q[18];
}
h q[19];
c[11] = measure q[19];
if (c[11] == 1) {
  h q[19];
  s q[19];
  s q[19];
  h q[19];
}
barrier q;
// correct_helpers
if (c[10] == 1) {
  h q[6];
  cx q[0], q[6];
  h q[6];
}
if (c[11] == 1) {
  h q[7];
  cx q[0], q[7];
  h q[7];
}
