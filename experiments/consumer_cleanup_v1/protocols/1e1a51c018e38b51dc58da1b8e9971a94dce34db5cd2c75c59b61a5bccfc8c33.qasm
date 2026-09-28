OPENQASM 3.0;
include "stdgates.inc";
qubit[31] q;
bit[21] c;
// prepare
h q[28];
t q[28];
cx q[7], q[28];
tdg q[28];
cx q[0], q[28];
t q[28];
cx q[7], q[28];
tdg q[28];
h q[28];
sdg q[28];
h q[29];
t q[29];
cx q[8], q[29];
tdg q[29];
cx q[0], q[29];
t q[29];
cx q[8], q[29];
tdg q[29];
h q[29];
sdg q[29];
h q[30];
t q[30];
cx q[9], q[30];
tdg q[30];
cx q[0], q[30];
t q[30];
cx q[9], q[30];
tdg q[30];
h q[30];
sdg q[30];
h q[10];
t q[10];
cx q[28], q[10];
tdg q[10];
cx q[1], q[10];
t q[10];
cx q[28], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[14];
t q[14];
cx q[29], q[14];
tdg q[14];
cx q[2], q[14];
t q[14];
cx q[29], q[14];
tdg q[14];
h q[14];
sdg q[14];
h q[18];
t q[18];
cx q[30], q[18];
tdg q[18];
cx q[3], q[18];
t q[18];
cx q[30], q[18];
tdg q[18];
h q[18];
sdg q[18];
h q[13];
t q[13];
cx q[28], q[13];
tdg q[13];
cx q[2], q[13];
t q[13];
cx q[28], q[13];
tdg q[13];
h q[13];
sdg q[13];
h q[16];
t q[16];
cx q[28], q[16];
tdg q[16];
cx q[3], q[16];
t q[16];
cx q[28], q[16];
tdg q[16];
h q[16];
sdg q[16];
h q[19];
t q[19];
cx q[28], q[19];
tdg q[19];
cx q[4], q[19];
t q[19];
cx q[28], q[19];
tdg q[19];
h q[19];
sdg q[19];
h q[22];
t q[22];
cx q[28], q[22];
tdg q[22];
cx q[5], q[22];
t q[22];
cx q[28], q[22];
tdg q[22];
h q[22];
sdg q[22];
h q[25];
t q[25];
cx q[28], q[25];
tdg q[25];
cx q[6], q[25];
t q[25];
cx q[28], q[25];
tdg q[25];
h q[25];
sdg q[25];
h q[26];
t q[26];
cx q[29], q[26];
tdg q[26];
cx q[6], q[26];
t q[26];
cx q[29], q[26];
tdg q[26];
h q[26];
sdg q[26];
h q[11];
t q[11];
cx q[29], q[11];
tdg q[11];
cx q[1], q[11];
t q[11];
cx q[29], q[11];
tdg q[11];
h q[11];
sdg q[11];
h q[17];
t q[17];
cx q[29], q[17];
tdg q[17];
cx q[3], q[17];
t q[17];
cx q[29], q[17];
tdg q[17];
h q[17];
sdg q[17];
h q[20];
t q[20];
cx q[29], q[20];
tdg q[20];
cx q[4], q[20];
t q[20];
cx q[29], q[20];
tdg q[20];
h q[20];
sdg q[20];
h q[23];
t q[23];
cx q[29], q[23];
tdg q[23];
cx q[5], q[23];
t q[23];
cx q[29], q[23];
tdg q[23];
h q[23];
sdg q[23];
h q[24];
t q[24];
cx q[30], q[24];
tdg q[24];
cx q[5], q[24];
t q[24];
cx q[30], q[24];
tdg q[24];
h q[24];
sdg q[24];
h q[12];
t q[12];
cx q[30], q[12];
tdg q[12];
cx q[1], q[12];
t q[12];
cx q[30], q[12];
tdg q[12];
h q[12];
sdg q[12];
h q[15];
t q[15];
cx q[30], q[15];
tdg q[15];
cx q[2], q[15];
t q[15];
cx q[30], q[15];
tdg q[15];
h q[15];
sdg q[15];
h q[21];
t q[21];
cx q[30], q[21];
tdg q[21];
cx q[4], q[21];
t q[21];
cx q[30], q[21];
tdg q[21];
h q[21];
sdg q[21];
h q[27];
t q[27];
cx q[30], q[27];
tdg q[27];
cx q[6], q[27];
t q[27];
cx q[30], q[27];
tdg q[27];
h q[27];
sdg q[27];
barrier q;
// consumer
h q[7];
cx q[0], q[7];
h q[7];
h q[23];
cx q[1], q[23];
h q[23];
h q[27];
cx q[2], q[27];
h q[27];
h q[11];
cx q[3], q[11];
h q[11];
h q[15];
cx q[4], q[15];
h q[15];
h q[19];
cx q[6], q[19];
h q[19];
h q[18];
cx q[10], q[18];
h q[18];
h q[23];
cx q[10], q[23];
h q[23];
h q[16];
cx q[11], q[16];
h q[16];
h q[24];
cx q[11], q[24];
h q[24];
h q[17];
cx q[12], q[17];
h q[17];
h q[22];
cx q[12], q[22];
h q[22];
h q[21];
cx q[13], q[21];
h q[21];
h q[26];
cx q[13], q[26];
h q[26];
h q[19];
cx q[14], q[19];
h q[19];
h q[27];
cx q[14], q[27];
h q[27];
h q[20];
cx q[15], q[20];
h q[20];
h q[25];
cx q[15], q[25];
h q[25];
h q[24];
cx q[16], q[24];
h q[24];
h q[22];
cx q[17], q[22];
h q[22];
h q[23];
cx q[18], q[23];
h q[23];
h q[27];
cx q[19], q[27];
h q[27];
h q[25];
cx q[20], q[25];
h q[25];
h q[26];
cx q[21], q[26];
h q[26];
s q[10];
s q[10];
s q[13];
s q[13];
s q[16];
s q[16];
s q[19];
s q[19];
s q[22];
s q[22];
s q[25];
s q[25];
barrier q;
// measure_products
h q[10];
c[0] = measure q[10];
if (c[0] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
h q[11];
c[1] = measure q[11];
if (c[1] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
h q[12];
c[2] = measure q[12];
if (c[2] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
h q[13];
c[3] = measure q[13];
if (c[3] == 1) {
  h q[13];
  s q[13];
  s q[13];
  h q[13];
}
h q[14];
c[4] = measure q[14];
if (c[4] == 1) {
  h q[14];
  s q[14];
  s q[14];
  h q[14];
}
h q[15];
c[5] = measure q[15];
if (c[5] == 1) {
  h q[15];
  s q[15];
  s q[15];
  h q[15];
}
h q[16];
c[6] = measure q[16];
if (c[6] == 1) {
  h q[16];
  s q[16];
  s q[16];
  h q[16];
}
h q[17];
c[7] = measure q[17];
if (c[7] == 1) {
  h q[17];
  s q[17];
  s q[17];
  h q[17];
}
h q[18];
c[8] = measure q[18];
if (c[8] == 1) {
  h q[18];
  s q[18];
  s q[18];
  h q[18];
}
h q[19];
c[9] = measure q[19];
if (c[9] == 1) {
  h q[19];
  s q[19];
  s q[19];
  h q[19];
}
h q[20];
c[10] = measure q[20];
if (c[10] == 1) {
  h q[20];
  s q[20];
  s q[20];
  h q[20];
}
h q[21];
c[11] = measure q[21];
if (c[11] == 1) {
  h q[21];
  s q[21];
  s q[21];
  h q[21];
}
h q[22];
c[12] = measure q[22];
if (c[12] == 1) {
  h q[22];
  s q[22];
  s q[22];
  h q[22];
}
h q[23];
c[13] = measure q[23];
if (c[13] == 1) {
  h q[23];
  s q[23];
  s q[23];
  h q[23];
}
h q[24];
c[14] = measure q[24];
if (c[14] == 1) {
  h q[24];
  s q[24];
  s q[24];
  h q[24];
}
h q[25];
c[15] = measure q[25];
if (c[15] == 1) {
  h q[25];
  s q[25];
  s q[25];
  h q[25];
}
h q[26];
c[16] = measure q[26];
if (c[16] == 1) {
  h q[26];
  s q[26];
  s q[26];
  h q[26];
}
h q[27];
c[17] = measure q[27];
if (c[17] == 1) {
  h q[27];
  s q[27];
  s q[27];
  h q[27];
}
barrier q;
// correct_products
if (c[0] == 1) {
  h q[28];
  cx q[1], q[28];
  h q[28];
}
if (c[1] == 1) {
  h q[29];
  cx q[1], q[29];
  h q[29];
}
if (c[2] == 1) {
  h q[30];
  cx q[1], q[30];
  h q[30];
}
if (c[3] == 1) {
  h q[28];
  cx q[2], q[28];
  h q[28];
}
if (c[4] == 1) {
  h q[29];
  cx q[2], q[29];
  h q[29];
}
if (c[5] == 1) {
  h q[30];
  cx q[2], q[30];
  h q[30];
}
if (c[6] == 1) {
  h q[28];
  cx q[3], q[28];
  h q[28];
}
if (c[7] == 1) {
  h q[29];
  cx q[3], q[29];
  h q[29];
}
if (c[8] == 1) {
  h q[30];
  cx q[3], q[30];
  h q[30];
}
if (c[9] == 1) {
  h q[28];
  cx q[4], q[28];
  h q[28];
}
if (c[10] == 1) {
  h q[29];
  cx q[4], q[29];
  h q[29];
}
if (c[11] == 1) {
  h q[30];
  cx q[4], q[30];
  h q[30];
}
if (c[12] == 1) {
  h q[28];
  cx q[5], q[28];
  h q[28];
}
if (c[13] == 1) {
  h q[29];
  cx q[5], q[29];
  h q[29];
}
if (c[14] == 1) {
  h q[30];
  cx q[5], q[30];
  h q[30];
}
if (c[15] == 1) {
  h q[28];
  cx q[6], q[28];
  h q[28];
}
if (c[16] == 1) {
  h q[29];
  cx q[6], q[29];
  h q[29];
}
if (c[17] == 1) {
  h q[30];
  cx q[6], q[30];
  h q[30];
}
barrier q;
// measure_helpers
h q[28];
c[18] = measure q[28];
if (c[18] == 1) {
  h q[28];
  s q[28];
  s q[28];
  h q[28];
}
h q[29];
c[19] = measure q[29];
if (c[19] == 1) {
  h q[29];
  s q[29];
  s q[29];
  h q[29];
}
h q[30];
c[20] = measure q[30];
if (c[20] == 1) {
  h q[30];
  s q[30];
  s q[30];
  h q[30];
}
barrier q;
// correct_helpers
if (c[18] == 1) {
  h q[7];
  cx q[0], q[7];
  h q[7];
}
if (c[19] == 1) {
  h q[8];
  cx q[0], q[8];
  h q[8];
}
if (c[20] == 1) {
  h q[9];
  cx q[0], q[9];
  h q[9];
}
