OPENQASM 3.0;
include "stdgates.inc";
qubit[44] q;
bit[32] c;
// prepare
h q[36];
t q[36];
cx q[1], q[36];
tdg q[36];
cx q[0], q[36];
t q[36];
cx q[1], q[36];
tdg q[36];
h q[36];
sdg q[36];
h q[37];
t q[37];
cx q[2], q[37];
tdg q[37];
cx q[0], q[37];
t q[37];
cx q[2], q[37];
tdg q[37];
h q[37];
sdg q[37];
h q[38];
t q[38];
cx q[3], q[38];
tdg q[38];
cx q[0], q[38];
t q[38];
cx q[3], q[38];
tdg q[38];
h q[38];
sdg q[38];
h q[39];
t q[39];
cx q[4], q[39];
tdg q[39];
cx q[0], q[39];
t q[39];
cx q[4], q[39];
tdg q[39];
h q[39];
sdg q[39];
h q[40];
t q[40];
cx q[5], q[40];
tdg q[40];
cx q[0], q[40];
t q[40];
cx q[5], q[40];
tdg q[40];
h q[40];
sdg q[40];
h q[41];
t q[41];
cx q[6], q[41];
tdg q[41];
cx q[0], q[41];
t q[41];
cx q[6], q[41];
tdg q[41];
h q[41];
sdg q[41];
h q[42];
t q[42];
cx q[7], q[42];
tdg q[42];
cx q[0], q[42];
t q[42];
cx q[7], q[42];
tdg q[42];
h q[42];
sdg q[42];
h q[43];
t q[43];
cx q[8], q[43];
tdg q[43];
cx q[0], q[43];
t q[43];
cx q[8], q[43];
tdg q[43];
h q[43];
sdg q[43];
h q[33];
t q[33];
cx q[9], q[33];
tdg q[33];
cx q[43], q[33];
t q[33];
cx q[9], q[33];
tdg q[33];
h q[33];
sdg q[33];
h q[12];
t q[12];
cx q[9], q[12];
tdg q[12];
cx q[36], q[12];
t q[12];
cx q[9], q[12];
tdg q[12];
h q[12];
sdg q[12];
h q[15];
t q[15];
cx q[9], q[15];
tdg q[15];
cx q[37], q[15];
t q[15];
cx q[9], q[15];
tdg q[15];
h q[15];
sdg q[15];
h q[18];
t q[18];
cx q[9], q[18];
tdg q[18];
cx q[38], q[18];
t q[18];
cx q[9], q[18];
tdg q[18];
h q[18];
sdg q[18];
h q[21];
t q[21];
cx q[9], q[21];
tdg q[21];
cx q[39], q[21];
t q[21];
cx q[9], q[21];
tdg q[21];
h q[21];
sdg q[21];
h q[24];
t q[24];
cx q[9], q[24];
tdg q[24];
cx q[40], q[24];
t q[24];
cx q[9], q[24];
tdg q[24];
h q[24];
sdg q[24];
h q[27];
t q[27];
cx q[9], q[27];
tdg q[27];
cx q[41], q[27];
t q[27];
cx q[9], q[27];
tdg q[27];
h q[27];
sdg q[27];
h q[30];
t q[30];
cx q[9], q[30];
tdg q[30];
cx q[42], q[30];
t q[30];
cx q[9], q[30];
tdg q[30];
h q[30];
sdg q[30];
h q[31];
t q[31];
cx q[10], q[31];
tdg q[31];
cx q[42], q[31];
t q[31];
cx q[10], q[31];
tdg q[31];
h q[31];
sdg q[31];
h q[13];
t q[13];
cx q[10], q[13];
tdg q[13];
cx q[36], q[13];
t q[13];
cx q[10], q[13];
tdg q[13];
h q[13];
sdg q[13];
h q[16];
t q[16];
cx q[10], q[16];
tdg q[16];
cx q[37], q[16];
t q[16];
cx q[10], q[16];
tdg q[16];
h q[16];
sdg q[16];
h q[19];
t q[19];
cx q[10], q[19];
tdg q[19];
cx q[38], q[19];
t q[19];
cx q[10], q[19];
tdg q[19];
h q[19];
sdg q[19];
h q[22];
t q[22];
cx q[10], q[22];
tdg q[22];
cx q[39], q[22];
t q[22];
cx q[10], q[22];
tdg q[22];
h q[22];
sdg q[22];
h q[25];
t q[25];
cx q[10], q[25];
tdg q[25];
cx q[40], q[25];
t q[25];
cx q[10], q[25];
tdg q[25];
h q[25];
sdg q[25];
h q[28];
t q[28];
cx q[10], q[28];
tdg q[28];
cx q[41], q[28];
t q[28];
cx q[10], q[28];
tdg q[28];
h q[28];
sdg q[28];
h q[34];
t q[34];
cx q[10], q[34];
tdg q[34];
cx q[43], q[34];
t q[34];
cx q[10], q[34];
tdg q[34];
h q[34];
sdg q[34];
h q[35];
t q[35];
cx q[11], q[35];
tdg q[35];
cx q[43], q[35];
t q[35];
cx q[11], q[35];
tdg q[35];
h q[35];
sdg q[35];
h q[14];
t q[14];
cx q[11], q[14];
tdg q[14];
cx q[36], q[14];
t q[14];
cx q[11], q[14];
tdg q[14];
h q[14];
sdg q[14];
h q[17];
t q[17];
cx q[11], q[17];
tdg q[17];
cx q[37], q[17];
t q[17];
cx q[11], q[17];
tdg q[17];
h q[17];
sdg q[17];
h q[20];
t q[20];
cx q[11], q[20];
tdg q[20];
cx q[38], q[20];
t q[20];
cx q[11], q[20];
tdg q[20];
h q[20];
sdg q[20];
h q[23];
t q[23];
cx q[11], q[23];
tdg q[23];
cx q[39], q[23];
t q[23];
cx q[11], q[23];
tdg q[23];
h q[23];
sdg q[23];
h q[26];
t q[26];
cx q[11], q[26];
tdg q[26];
cx q[40], q[26];
t q[26];
cx q[11], q[26];
tdg q[26];
h q[26];
sdg q[26];
h q[29];
t q[29];
cx q[11], q[29];
tdg q[29];
cx q[41], q[29];
t q[29];
cx q[11], q[29];
tdg q[29];
h q[29];
sdg q[29];
h q[32];
t q[32];
cx q[11], q[32];
tdg q[32];
cx q[42], q[32];
t q[32];
cx q[11], q[32];
tdg q[32];
h q[32];
sdg q[32];
barrier q;
// consumer
h q[32];
cx q[1], q[32];
h q[32];
h q[12];
cx q[3], q[12];
h q[12];
h q[16];
cx q[4], q[16];
h q[16];
h q[20];
cx q[5], q[20];
h q[20];
h q[24];
cx q[7], q[24];
h q[24];
h q[28];
cx q[8], q[28];
h q[28];
h q[23];
cx q[12], q[23];
h q[23];
h q[28];
cx q[12], q[28];
h q[28];
h q[21];
cx q[13], q[21];
h q[21];
h q[29];
cx q[13], q[29];
h q[29];
h q[22];
cx q[14], q[22];
h q[22];
h q[27];
cx q[14], q[27];
h q[27];
h q[26];
cx q[15], q[26];
h q[26];
h q[31];
cx q[15], q[31];
h q[31];
h q[24];
cx q[16], q[24];
h q[24];
h q[32];
cx q[16], q[32];
h q[32];
h q[25];
cx q[17], q[25];
h q[25];
h q[30];
cx q[17], q[30];
h q[30];
h q[29];
cx q[18], q[29];
h q[29];
h q[34];
cx q[18], q[34];
h q[34];
h q[27];
cx q[19], q[27];
h q[27];
h q[35];
cx q[19], q[35];
h q[35];
h q[28];
cx q[20], q[28];
h q[28];
h q[33];
cx q[20], q[33];
h q[33];
h q[32];
cx q[21], q[32];
h q[32];
h q[30];
cx q[22], q[30];
h q[30];
h q[31];
cx q[23], q[31];
h q[31];
h q[35];
cx q[24], q[35];
h q[35];
h q[33];
cx q[25], q[33];
h q[33];
h q[34];
cx q[26], q[34];
h q[34];
s q[14];
s q[14];
s q[17];
s q[17];
s q[20];
s q[20];
s q[23];
s q[23];
s q[26];
s q[26];
s q[29];
s q[29];
s q[32];
s q[32];
s q[35];
s q[35];
barrier q;
// measure_products
h q[12];
c[0] = measure q[12];
if (c[0] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
h q[13];
c[1] = measure q[13];
if (c[1] == 1) {
  h q[13];
  s q[13];
  s q[13];
  h q[13];
}
h q[14];
c[2] = measure q[14];
if (c[2] == 1) {
  h q[14];
  s q[14];
  s q[14];
  h q[14];
}
h q[15];
c[3] = measure q[15];
if (c[3] == 1) {
  h q[15];
  s q[15];
  s q[15];
  h q[15];
}
h q[16];
c[4] = measure q[16];
if (c[4] == 1) {
  h q[16];
  s q[16];
  s q[16];
  h q[16];
}
h q[17];
c[5] = measure q[17];
if (c[5] == 1) {
  h q[17];
  s q[17];
  s q[17];
  h q[17];
}
h q[18];
c[6] = measure q[18];
if (c[6] == 1) {
  h q[18];
  s q[18];
  s q[18];
  h q[18];
}
h q[19];
c[7] = measure q[19];
if (c[7] == 1) {
  h q[19];
  s q[19];
  s q[19];
  h q[19];
}
h q[20];
c[8] = measure q[20];
if (c[8] == 1) {
  h q[20];
  s q[20];
  s q[20];
  h q[20];
}
h q[21];
c[9] = measure q[21];
if (c[9] == 1) {
  h q[21];
  s q[21];
  s q[21];
  h q[21];
}
h q[22];
c[10] = measure q[22];
if (c[10] == 1) {
  h q[22];
  s q[22];
  s q[22];
  h q[22];
}
h q[23];
c[11] = measure q[23];
if (c[11] == 1) {
  h q[23];
  s q[23];
  s q[23];
  h q[23];
}
h q[24];
c[12] = measure q[24];
if (c[12] == 1) {
  h q[24];
  s q[24];
  s q[24];
  h q[24];
}
h q[25];
c[13] = measure q[25];
if (c[13] == 1) {
  h q[25];
  s q[25];
  s q[25];
  h q[25];
}
h q[26];
c[14] = measure q[26];
if (c[14] == 1) {
  h q[26];
  s q[26];
  s q[26];
  h q[26];
}
h q[27];
c[15] = measure q[27];
if (c[15] == 1) {
  h q[27];
  s q[27];
  s q[27];
  h q[27];
}
h q[28];
c[16] = measure q[28];
if (c[16] == 1) {
  h q[28];
  s q[28];
  s q[28];
  h q[28];
}
h q[29];
c[17] = measure q[29];
if (c[17] == 1) {
  h q[29];
  s q[29];
  s q[29];
  h q[29];
}
h q[30];
c[18] = measure q[30];
if (c[18] == 1) {
  h q[30];
  s q[30];
  s q[30];
  h q[30];
}
h q[31];
c[19] = measure q[31];
if (c[19] == 1) {
  h q[31];
  s q[31];
  s q[31];
  h q[31];
}
h q[32];
c[20] = measure q[32];
if (c[20] == 1) {
  h q[32];
  s q[32];
  s q[32];
  h q[32];
}
h q[33];
c[21] = measure q[33];
if (c[21] == 1) {
  h q[33];
  s q[33];
  s q[33];
  h q[33];
}
h q[34];
c[22] = measure q[34];
if (c[22] == 1) {
  h q[34];
  s q[34];
  s q[34];
  h q[34];
}
h q[35];
c[23] = measure q[35];
if (c[23] == 1) {
  h q[35];
  s q[35];
  s q[35];
  h q[35];
}
barrier q;
// correct_products
if (c[0] == 1) {
  h q[9];
  cx q[36], q[9];
  h q[9];
}
if (c[1] == 1) {
  h q[10];
  cx q[36], q[10];
  h q[10];
}
if (c[2] == 1) {
  h q[11];
  cx q[36], q[11];
  h q[11];
}
if (c[3] == 1) {
  h q[9];
  cx q[37], q[9];
  h q[9];
}
if (c[4] == 1) {
  h q[10];
  cx q[37], q[10];
  h q[10];
}
if (c[5] == 1) {
  h q[11];
  cx q[37], q[11];
  h q[11];
}
if (c[6] == 1) {
  h q[9];
  cx q[38], q[9];
  h q[9];
}
if (c[7] == 1) {
  h q[10];
  cx q[38], q[10];
  h q[10];
}
if (c[8] == 1) {
  h q[11];
  cx q[38], q[11];
  h q[11];
}
if (c[9] == 1) {
  h q[9];
  cx q[39], q[9];
  h q[9];
}
if (c[10] == 1) {
  h q[10];
  cx q[39], q[10];
  h q[10];
}
if (c[11] == 1) {
  h q[11];
  cx q[39], q[11];
  h q[11];
}
if (c[12] == 1) {
  h q[9];
  cx q[40], q[9];
  h q[9];
}
if (c[13] == 1) {
  h q[10];
  cx q[40], q[10];
  h q[10];
}
if (c[14] == 1) {
  h q[11];
  cx q[40], q[11];
  h q[11];
}
if (c[15] == 1) {
  h q[9];
  cx q[41], q[9];
  h q[9];
}
if (c[16] == 1) {
  h q[10];
  cx q[41], q[10];
  h q[10];
}
if (c[17] == 1) {
  h q[11];
  cx q[41], q[11];
  h q[11];
}
if (c[18] == 1) {
  h q[9];
  cx q[42], q[9];
  h q[9];
}
if (c[19] == 1) {
  h q[10];
  cx q[42], q[10];
  h q[10];
}
if (c[20] == 1) {
  h q[11];
  cx q[42], q[11];
  h q[11];
}
if (c[21] == 1) {
  h q[9];
  cx q[43], q[9];
  h q[9];
}
if (c[22] == 1) {
  h q[10];
  cx q[43], q[10];
  h q[10];
}
if (c[23] == 1) {
  h q[11];
  cx q[43], q[11];
  h q[11];
}
barrier q;
// measure_helpers
h q[36];
c[24] = measure q[36];
if (c[24] == 1) {
  h q[36];
  s q[36];
  s q[36];
  h q[36];
}
h q[37];
c[25] = measure q[37];
if (c[25] == 1) {
  h q[37];
  s q[37];
  s q[37];
  h q[37];
}
h q[38];
c[26] = measure q[38];
if (c[26] == 1) {
  h q[38];
  s q[38];
  s q[38];
  h q[38];
}
h q[39];
c[27] = measure q[39];
if (c[27] == 1) {
  h q[39];
  s q[39];
  s q[39];
  h q[39];
}
h q[40];
c[28] = measure q[40];
if (c[28] == 1) {
  h q[40];
  s q[40];
  s q[40];
  h q[40];
}
h q[41];
c[29] = measure q[41];
if (c[29] == 1) {
  h q[41];
  s q[41];
  s q[41];
  h q[41];
}
h q[42];
c[30] = measure q[42];
if (c[30] == 1) {
  h q[42];
  s q[42];
  s q[42];
  h q[42];
}
h q[43];
c[31] = measure q[43];
if (c[31] == 1) {
  h q[43];
  s q[43];
  s q[43];
  h q[43];
}
barrier q;
// correct_helpers
if (c[24] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
if (c[25] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
if (c[26] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
if (c[27] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
if (c[28] == 1) {
  h q[5];
  cx q[0], q[5];
  h q[5];
}
if (c[29] == 1) {
  h q[6];
  cx q[0], q[6];
  h q[6];
}
if (c[30] == 1) {
  h q[7];
  cx q[0], q[7];
  h q[7];
}
if (c[31] == 1) {
  h q[8];
  cx q[0], q[8];
  h q[8];
}
