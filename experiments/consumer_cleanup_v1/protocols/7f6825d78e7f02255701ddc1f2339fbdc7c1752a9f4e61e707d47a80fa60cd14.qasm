OPENQASM 3.0;
include "stdgates.inc";
qubit[8] q;
bit[2] c;
// prepare
h q[7];
t q[7];
cx q[2], q[7];
tdg q[7];
cx q[0], q[7];
t q[7];
cx q[2], q[7];
tdg q[7];
h q[7];
sdg q[7];
barrier q;
// consumer
h q[5];
cx q[7], q[5];
h q[5];
barrier q;
// measure_reference_0
h q[7];
c[0] = measure q[7];
if (c[0] == 1) {
  h q[7];
  s q[7];
  s q[7];
  h q[7];
}
barrier q;
// correct_reference_0
if (c[0] == 1) {
  h q[2];
  cx q[0], q[2];
  h q[2];
}
barrier q;
// consumer
h q[6];
cx q[0], q[6];
h q[6];
barrier q;
// prepare
h q[7];
t q[7];
cx q[3], q[7];
tdg q[7];
cx q[0], q[7];
t q[7];
cx q[3], q[7];
tdg q[7];
h q[7];
sdg q[7];
barrier q;
// consumer
h q[6];
cx q[7], q[6];
h q[6];
barrier q;
// measure_reference_1
h q[7];
c[1] = measure q[7];
if (c[1] == 1) {
  h q[7];
  s q[7];
  s q[7];
  h q[7];
}
barrier q;
// correct_reference_1
if (c[1] == 1) {
  h q[3];
  cx q[0], q[3];
  h q[3];
}
