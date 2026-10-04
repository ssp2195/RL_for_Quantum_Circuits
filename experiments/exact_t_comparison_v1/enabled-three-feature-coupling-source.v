module oracle(x0,x1,x2,x3,x4,y);
input x0,x1,x2,x3,x4;
output y;
wire g0,g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,g12;
assign g0 = x0 & x1;
assign g1 = g0 & x4;
assign g2 = x0 & x2;
assign g3 = g2 & x4;
assign g4 = x0 & x3;
assign g5 = g4 & x4;
assign g6 = g1 & g3;
assign g7 = g1 & g5;
assign g8 = g3 & g5;
assign g9 = 1'b0 ^ g1;
assign g10 = g9 ^ g6;
assign g11 = g10 ^ g7;
assign g12 = g11 ^ g8;
assign y = g12;
endmodule
