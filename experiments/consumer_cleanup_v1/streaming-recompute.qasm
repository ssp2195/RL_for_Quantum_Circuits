OPENQASM 3.0;
include "stdgates.inc";
qubit[12] q;
bit[12] c;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[1], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
cx q[3], q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
h q[11];
sdg q[11];
cx q[10], q[1];
cx q[11], q[1];
h q[9];
t q[9];
cx q[1], q[9];
tdg q[9];
cx q[0], q[9];
t q[9];
cx q[1], q[9];
tdg q[9];
h q[9];
sdg q[9];
barrier q;
// consumer
h q[5];
cx q[9], q[5];
h q[5];
barrier q;
// measure_factor_0
h q[9];
c[0] = measure q[9];
if (c[0] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
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
cx q[11], q[1];
cx q[10], q[1];
barrier q;
// measure_features_0
h q[10];
c[1] = measure q[10];
if (c[1] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
h q[11];
c[2] = measure q[11];
if (c[2] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_features_0
if (c[1] == 1) {
  h q[2];
  cx q[1], q[2];
  h q[2];
}
if (c[2] == 1) {
  h q[4];
  cx q[3], q[4];
  h q[4];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
cx q[1], q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
cx q[2], q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
h q[11];
sdg q[11];
cx q[10], q[2];
cx q[11], q[2];
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
barrier q;
// consumer
h q[6];
cx q[9], q[6];
h q[6];
barrier q;
// measure_factor_1
h q[9];
c[3] = measure q[9];
if (c[3] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
}
barrier q;
// correct_factor_1
if (c[3] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// restore_factor_1
cx q[11], q[2];
cx q[10], q[2];
barrier q;
// measure_features_1
h q[10];
c[4] = measure q[10];
if (c[4] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
h q[11];
c[5] = measure q[11];
if (c[5] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_features_1
if (c[4] == 1) {
  h q[3];
  cx q[1], q[3];
  h q[3];
}
if (c[5] == 1) {
  h q[4];
  cx q[2], q[4];
  h q[4];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
cx q[1], q[10];
t q[10];
cx q[2], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
cx q[3], q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
h q[11];
sdg q[11];
cx q[10], q[3];
cx q[11], q[3];
h q[9];
t q[9];
cx q[3], q[9];
tdg q[9];
cx q[0], q[9];
t q[9];
cx q[3], q[9];
tdg q[9];
h q[9];
sdg q[9];
barrier q;
// consumer
h q[7];
cx q[9], q[7];
h q[7];
barrier q;
// measure_factor_2
h q[9];
c[6] = measure q[9];
if (c[6] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
}
barrier q;
// correct_factor_2
if (c[6] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
barrier q;
// restore_factor_2
cx q[11], q[3];
cx q[10], q[3];
barrier q;
// measure_features_2
h q[10];
c[7] = measure q[10];
if (c[7] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
h q[11];
c[8] = measure q[11];
if (c[8] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_features_2
if (c[7] == 1) {
  h q[2];
  cx q[1], q[2];
  h q[2];
}
if (c[8] == 1) {
  h q[4];
  cx q[3], q[4];
  h q[4];
}
barrier q;
// prepare
h q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
cx q[1], q[10];
t q[10];
cx q[3], q[10];
tdg q[10];
h q[10];
sdg q[10];
h q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
cx q[2], q[11];
t q[11];
cx q[4], q[11];
tdg q[11];
h q[11];
sdg q[11];
cx q[10], q[4];
cx q[11], q[4];
h q[9];
t q[9];
cx q[4], q[9];
tdg q[9];
cx q[0], q[9];
t q[9];
cx q[4], q[9];
tdg q[9];
h q[9];
sdg q[9];
barrier q;
// consumer
h q[8];
cx q[9], q[8];
h q[8];
barrier q;
// measure_factor_3
h q[9];
c[9] = measure q[9];
if (c[9] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
}
barrier q;
// correct_factor_3
if (c[9] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
barrier q;
// restore_factor_3
cx q[11], q[4];
cx q[10], q[4];
barrier q;
// measure_features_3
h q[10];
c[10] = measure q[10];
if (c[10] == 1) {
  h q[10];
  s q[10];
  s q[10];
  h q[10];
}
h q[11];
c[11] = measure q[11];
if (c[11] == 1) {
  h q[11];
  s q[11];
  s q[11];
  h q[11];
}
barrier q;
// correct_features_3
if (c[10] == 1) {
  h q[3];
  cx q[1], q[3];
  h q[3];
}
if (c[11] == 1) {
  h q[4];
  cx q[2], q[4];
  h q[4];
}
