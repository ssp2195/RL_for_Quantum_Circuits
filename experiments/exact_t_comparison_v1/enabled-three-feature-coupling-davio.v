module oracle(x0,x1,x2,x3,x4,y);
input x0,x1,x2,x3,x4;
output y;
wire g0,g1,g2,g3,g4,g5,g6;
assign g0 = x2 & x3;
assign g1 = ~1'b0 ^ x3;
assign g2 = g1 ^ x2;
assign g3 = x1 & g2;
assign g4 = g0 ^ g3;
assign g5 = x4 & g4;
assign g6 = x0 & g5;
assign y = g6;
endmodule
