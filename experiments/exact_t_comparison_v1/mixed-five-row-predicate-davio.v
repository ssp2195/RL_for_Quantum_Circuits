module oracle(x0,x1,x2,x3,x4,x5,x6,x7,y);
input x0,x1,x2,x3,x4,x5,x6,x7;
output y;
wire g0,g1,g2,g3,g4,g5,g6,g7,g8;
assign g0 = x3 & x6;
assign g1 = x1 & x7;
assign g2 = g0 ^ g1;
assign g3 = x2 ^ x1;
assign g4 = x6 & g3;
assign g5 = x7 ^ g4;
assign g6 = x4 & g5;
assign g7 = g2 ^ g6;
assign g8 = x0 & g7;
assign y = g8;
endmodule
