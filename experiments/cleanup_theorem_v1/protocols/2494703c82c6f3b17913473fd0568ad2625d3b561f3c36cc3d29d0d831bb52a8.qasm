OPENQASM 3.0;
include "stdgates.inc";
qubit[7] q;
bit[7] c;
// prepare
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
// measure_reference_0
h q[5];
c[0] = measure q[5];
if (c[0] == 1) {
  h q[5];
  s q[5];
  s q[5];
  h q[5];
}
barrier q;
// correct_reference_0
if (c[0] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
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
h q[6];
t q[6];
cx q[2], q[6];
tdg q[6];
cx q[5], q[6];
t q[6];
cx q[2], q[6];
tdg q[6];
h q[6];
sdg q[6];
barrier q;
// consumer
h q[4];
cx q[6], q[4];
h q[4];
barrier q;
// measure_reference_1
h q[6];
c[1] = measure q[6];
if (c[1] == 1) {
  h q[6];
  s q[6];
  s q[6];
  h q[6];
}
barrier q;
// correct_reference_1
if (c[1] == 1) {
  h q[2];
  cx q[5], q[2];
  h q[2];
}
barrier q;
// measure_reference_2
h q[5];
c[2] = measure q[5];
if (c[2] == 1) {
  h q[5];
  s q[5];
  s q[5];
  h q[5];
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
h q[6];
t q[6];
cx q[3], q[6];
tdg q[6];
cx q[5], q[6];
t q[6];
cx q[3], q[6];
tdg q[6];
h q[6];
sdg q[6];
barrier q;
// consumer
h q[4];
cx q[6], q[4];
h q[4];
barrier q;
// measure_reference_3
h q[6];
c[3] = measure q[6];
if (c[3] == 1) {
  h q[6];
  s q[6];
  s q[6];
  h q[6];
}
barrier q;
// correct_reference_3
if (c[3] == 1) {
  h q[3];
  cx q[5], q[3];
  h q[3];
}
barrier q;
// measure_reference_4
h q[5];
c[4] = measure q[5];
if (c[4] == 1) {
  h q[5];
  s q[5];
  s q[5];
  h q[5];
}
barrier q;
// correct_reference_4
if (c[4] == 1) {
  h q[1];
  cx q[0], q[1];
  h q[1];
}
barrier q;
// prepare
h q[5];
t q[5];
cx q[2], q[5];
tdg q[5];
cx q[0], q[5];
t q[5];
cx q[2], q[5];
tdg q[5];
h q[5];
sdg q[5];
h q[6];
t q[6];
cx q[3], q[6];
tdg q[6];
cx q[5], q[6];
t q[6];
cx q[3], q[6];
tdg q[6];
h q[6];
sdg q[6];
barrier q;
// consumer
h q[4];
cx q[6], q[4];
h q[4];
barrier q;
// measure_reference_5
h q[6];
c[5] = measure q[6];
if (c[5] == 1) {
  h q[6];
  s q[6];
  s q[6];
  h q[6];
}
barrier q;
// correct_reference_5
if (c[5] == 1) {
  h q[3];
  cx q[5], q[3];
  h q[3];
}
barrier q;
// measure_reference_6
h q[5];
c[6] = measure q[5];
if (c[6] == 1) {
  h q[5];
  s q[5];
  s q[5];
  h q[5];
}
barrier q;
// correct_reference_6
if (c[6] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
