OPENQASM 3.0;
include "stdgates.inc";
qubit[24] q;
bit[21] c;
// prepare
h q[13];
t q[13];
cx q[7], q[13];
tdg q[13];
cx q[1], q[13];
t q[13];
cx q[7], q[13];
tdg q[13];
h q[13];
sdg q[13];
h q[14];
t q[14];
cx q[5], q[14];
tdg q[14];
cx q[3], q[14];
t q[14];
cx q[5], q[14];
tdg q[14];
h q[14];
sdg q[14];
cx q[2], q[1];
cx q[3], q[1];
cx q[4], q[1];
cx q[5], q[1];
cx q[6], q[1];
cx q[7], q[1];
cx q[8], q[1];
cx q[14], q[1];
cx q[13], q[1];
h q[12];
t q[12];
cx q[1], q[12];
tdg q[12];
cx q[0], q[12];
t q[12];
cx q[1], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[11];
cx q[12], q[11];
h q[11];
barrier q;
// measure_factor_0
h q[12];
c[0] = measure q[12];
if (c[0] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
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
cx q[14], q[1];
cx q[8], q[1];
cx q[7], q[1];
cx q[6], q[1];
cx q[5], q[1];
cx q[4], q[1];
cx q[3], q[1];
cx q[2], q[1];
barrier q;
// measure_features_0
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
barrier q;
// correct_features_0
if (c[1] == 1) {
  h q[7];
  cx q[1], q[7];
  h q[7];
}
if (c[2] == 1) {
  h q[5];
  cx q[3], q[5];
  h q[5];
}
barrier q;
// prepare
h q[13];
t q[13];
cx q[3], q[13];
tdg q[13];
cx q[1], q[13];
t q[13];
cx q[3], q[13];
tdg q[13];
h q[13];
sdg q[13];
h q[14];
t q[14];
cx q[7], q[14];
tdg q[14];
cx q[5], q[14];
t q[14];
cx q[7], q[14];
tdg q[14];
h q[14];
sdg q[14];
cx q[14], q[13];
h q[12];
t q[12];
cx q[13], q[12];
tdg q[12];
cx q[0], q[12];
t q[12];
cx q[13], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[9];
cx q[12], q[9];
h q[9];
barrier q;
// measure_factor_1
h q[12];
c[3] = measure q[12];
if (c[3] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_factor_1
if (c[3] == 1) {
  h q[13];
  cx q[0], q[13];
  h q[13];
}
barrier q;
// restore_factor_1
cx q[14], q[13];
barrier q;
// measure_features_1
h q[13];
c[4] = measure q[13];
if (c[4] == 1) {
  h q[13];
  s q[13];
  s q[13];
  h q[13];
}
h q[14];
c[5] = measure q[14];
if (c[5] == 1) {
  h q[14];
  s q[14];
  s q[14];
  h q[14];
}
barrier q;
// correct_features_1
if (c[4] == 1) {
  h q[3];
  cx q[1], q[3];
  h q[3];
}
if (c[5] == 1) {
  h q[7];
  cx q[5], q[7];
  h q[7];
}
barrier q;
// prepare
h q[13];
t q[13];
cx q[4], q[13];
tdg q[13];
cx q[1], q[13];
t q[13];
cx q[4], q[13];
tdg q[13];
h q[13];
sdg q[13];
h q[14];
t q[14];
cx q[6], q[14];
tdg q[14];
cx q[1], q[14];
t q[14];
cx q[6], q[14];
tdg q[14];
h q[14];
sdg q[14];
h q[15];
t q[15];
cx q[5], q[15];
tdg q[15];
cx q[2], q[15];
t q[15];
cx q[5], q[15];
tdg q[15];
h q[15];
sdg q[15];
h q[16];
t q[16];
cx q[7], q[16];
tdg q[16];
cx q[2], q[16];
t q[16];
cx q[7], q[16];
tdg q[16];
h q[16];
sdg q[16];
h q[17];
t q[17];
cx q[6], q[17];
tdg q[17];
cx q[3], q[17];
t q[17];
cx q[6], q[17];
tdg q[17];
h q[17];
sdg q[17];
h q[18];
t q[18];
cx q[8], q[18];
tdg q[18];
cx q[3], q[18];
t q[18];
cx q[8], q[18];
tdg q[18];
h q[18];
sdg q[18];
h q[19];
t q[19];
cx q[7], q[19];
tdg q[19];
cx q[4], q[19];
t q[19];
cx q[7], q[19];
tdg q[19];
h q[19];
sdg q[19];
h q[20];
t q[20];
cx q[8], q[20];
tdg q[20];
cx q[5], q[20];
t q[20];
cx q[8], q[20];
tdg q[20];
h q[20];
sdg q[20];
h q[21];
t q[21];
cx q[10], q[21];
tdg q[21];
cx q[9], q[21];
t q[21];
cx q[10], q[21];
tdg q[21];
h q[21];
sdg q[21];
h q[22];
t q[22];
cx q[11], q[22];
tdg q[22];
cx q[9], q[22];
t q[22];
cx q[11], q[22];
tdg q[22];
h q[22];
sdg q[22];
h q[23];
t q[23];
cx q[11], q[23];
tdg q[23];
cx q[10], q[23];
t q[23];
cx q[11], q[23];
tdg q[23];
h q[23];
sdg q[23];
cx q[15], q[13];
cx q[14], q[13];
cx q[17], q[13];
cx q[16], q[13];
cx q[19], q[13];
cx q[18], q[13];
cx q[20], q[13];
cx q[22], q[21];
cx q[23], q[21];
h q[12];
t q[12];
cx q[13], q[12];
tdg q[12];
cx q[0], q[12];
t q[12];
cx q[13], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[21];
cx q[12], q[21];
h q[21];
barrier q;
// measure_factor_2
h q[12];
c[6] = measure q[12];
if (c[6] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_factor_2
if (c[6] == 1) {
  h q[13];
  cx q[0], q[13];
  h q[13];
}
barrier q;
// restore_factor_2
cx q[23], q[21];
cx q[22], q[21];
cx q[20], q[13];
cx q[18], q[13];
cx q[19], q[13];
cx q[16], q[13];
cx q[17], q[13];
cx q[14], q[13];
cx q[15], q[13];
barrier q;
// measure_features_2
h q[13];
c[7] = measure q[13];
if (c[7] == 1) {
  h q[13];
  s q[13];
  s q[13];
  h q[13];
}
h q[14];
c[8] = measure q[14];
if (c[8] == 1) {
  h q[14];
  s q[14];
  s q[14];
  h q[14];
}
h q[15];
c[9] = measure q[15];
if (c[9] == 1) {
  h q[15];
  s q[15];
  s q[15];
  h q[15];
}
h q[16];
c[10] = measure q[16];
if (c[10] == 1) {
  h q[16];
  s q[16];
  s q[16];
  h q[16];
}
h q[17];
c[11] = measure q[17];
if (c[11] == 1) {
  h q[17];
  s q[17];
  s q[17];
  h q[17];
}
h q[18];
c[12] = measure q[18];
if (c[12] == 1) {
  h q[18];
  s q[18];
  s q[18];
  h q[18];
}
h q[19];
c[13] = measure q[19];
if (c[13] == 1) {
  h q[19];
  s q[19];
  s q[19];
  h q[19];
}
h q[20];
c[14] = measure q[20];
if (c[14] == 1) {
  h q[20];
  s q[20];
  s q[20];
  h q[20];
}
h q[21];
c[15] = measure q[21];
if (c[15] == 1) {
  h q[21];
  s q[21];
  s q[21];
  h q[21];
}
h q[22];
c[16] = measure q[22];
if (c[16] == 1) {
  h q[22];
  s q[22];
  s q[22];
  h q[22];
}
h q[23];
c[17] = measure q[23];
if (c[17] == 1) {
  h q[23];
  s q[23];
  s q[23];
  h q[23];
}
barrier q;
// correct_features_2
if (c[7] == 1) {
  h q[4];
  cx q[1], q[4];
  h q[4];
}
if (c[8] == 1) {
  h q[6];
  cx q[1], q[6];
  h q[6];
}
if (c[9] == 1) {
  h q[5];
  cx q[2], q[5];
  h q[5];
}
if (c[10] == 1) {
  h q[7];
  cx q[2], q[7];
  h q[7];
}
if (c[11] == 1) {
  h q[6];
  cx q[3], q[6];
  h q[6];
}
if (c[12] == 1) {
  h q[8];
  cx q[3], q[8];
  h q[8];
}
if (c[13] == 1) {
  h q[7];
  cx q[4], q[7];
  h q[7];
}
if (c[14] == 1) {
  h q[8];
  cx q[5], q[8];
  h q[8];
}
if (c[15] == 1) {
  h q[10];
  cx q[9], q[10];
  h q[10];
}
if (c[16] == 1) {
  h q[11];
  cx q[9], q[11];
  h q[11];
}
if (c[17] == 1) {
  h q[11];
  cx q[10], q[11];
  h q[11];
}
barrier q;
// prepare
h q[13];
t q[13];
cx q[4], q[13];
tdg q[13];
cx q[2], q[13];
t q[13];
cx q[4], q[13];
tdg q[13];
h q[13];
sdg q[13];
h q[14];
t q[14];
cx q[8], q[14];
tdg q[14];
cx q[6], q[14];
t q[14];
cx q[8], q[14];
tdg q[14];
h q[14];
sdg q[14];
cx q[14], q[13];
h q[12];
t q[12];
cx q[13], q[12];
tdg q[12];
cx q[0], q[12];
t q[12];
cx q[13], q[12];
tdg q[12];
h q[12];
sdg q[12];
barrier q;
// consumer
h q[10];
cx q[12], q[10];
h q[10];
barrier q;
// measure_factor_3
h q[12];
c[18] = measure q[12];
if (c[18] == 1) {
  h q[12];
  s q[12];
  s q[12];
  h q[12];
}
barrier q;
// correct_factor_3
if (c[18] == 1) {
  h q[13];
  cx q[0], q[13];
  h q[13];
}
barrier q;
// restore_factor_3
cx q[14], q[13];
barrier q;
// measure_features_3
h q[13];
c[19] = measure q[13];
if (c[19] == 1) {
  h q[13];
  s q[13];
  s q[13];
  h q[13];
}
h q[14];
c[20] = measure q[14];
if (c[20] == 1) {
  h q[14];
  s q[14];
  s q[14];
  h q[14];
}
barrier q;
// correct_features_3
if (c[19] == 1) {
  h q[4];
  cx q[2], q[4];
  h q[4];
}
if (c[20] == 1) {
  h q[8];
  cx q[6], q[8];
  h q[8];
}
