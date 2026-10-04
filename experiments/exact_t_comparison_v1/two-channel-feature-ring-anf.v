module oracle(x0,x1,x2,x3,x4,x5,x6,y);
input x0,x1,x2,x3,x4,x5,x6;
output y;
wire g0,g1,g2,g3,g4,g5,g6,g7;
assign g0 = x0 & x2;
assign g1 = g0 & x5;
assign g2 = x0 & x6;
assign g3 = x0 & x3;
assign g4 = g3 & x6;
assign g5 = 1'b0 ^ g1;
assign g6 = g5 ^ g2;
assign g7 = g6 ^ g4;
assign y = g7;
endmodule
