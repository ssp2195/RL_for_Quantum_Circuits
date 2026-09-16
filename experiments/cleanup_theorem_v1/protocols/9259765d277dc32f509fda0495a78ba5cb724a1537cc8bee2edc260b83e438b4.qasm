OPENQASM 3.0;
include "stdgates.inc";
qubit[10] q;
bit[7] c;
// prepare
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
// measure_reference_0
h q[8];
c[0] = measure q[8];
if (c[0] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
barrier q;
// correct_reference_0
if (c[0] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
barrier q;
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
cx q[4], q[9];
tdg q[9];
cx q[8], q[9];
t q[9];
cx q[4], q[9];
tdg q[9];
h q[9];
sdg q[9];
barrier q;
// consumer
h q[6];
cx q[9], q[6];
h q[6];
barrier q;
// measure_reference_1
h q[9];
c[1] = measure q[9];
if (c[1] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
}
barrier q;
// correct_reference_1
if (c[1] == 1) {
  h q[4];
  cx q[8], q[4];
  h q[4];
}
barrier q;
// measure_reference_2
h q[8];
c[2] = measure q[8];
if (c[2] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
barrier q;
// correct_reference_2
if (c[2] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[8];
t q[8];
cx q[2], q[8];
tdg q[8];
cx q[0], q[8];
t q[8];
cx q[2], q[8];
tdg q[8];
h q[8];
sdg q[8];
h q[9];
t q[9];
cx q[4], q[9];
tdg q[9];
cx q[8], q[9];
t q[9];
cx q[4], q[9];
tdg q[9];
h q[9];
sdg q[9];
barrier q;
// consumer
h q[6];
cx q[9], q[6];
h q[6];
barrier q;
// measure_reference_3
h q[9];
c[3] = measure q[9];
if (c[3] == 1) {
  h q[9];
  s q[9];
  s q[9];
  h q[9];
}
barrier q;
// correct_reference_3
if (c[3] == 1) {
  h q[4];
  cx q[8], q[4];
  h q[4];
}
barrier q;
// measure_reference_4
h q[8];
c[4] = measure q[8];
if (c[4] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
barrier q;
// correct_reference_4
if (c[4] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
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
barrier q;
// consumer
h q[7];
cx q[8], q[7];
h q[7];
barrier q;
// measure_reference_5
h q[8];
c[5] = measure q[8];
if (c[5] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
barrier q;
// correct_reference_5
if (c[5] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
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
barrier q;
// consumer
h q[7];
cx q[8], q[7];
h q[7];
barrier q;
// measure_reference_6
h q[8];
c[6] = measure q[8];
if (c[6] == 1) {
  h q[8];
  s q[8];
  s q[8];
  h q[8];
}
barrier q;
// correct_reference_6
if (c[6] == 1) {
  h q[4];
  cx q[0], q[4];
  h q[4];
}
