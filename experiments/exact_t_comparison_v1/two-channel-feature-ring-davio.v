module oracle(x0,x1,x2,x3,x4,x5,x6,y);
input x0,x1,x2,x3,x4,x5,x6;
output y;
wire g0,g1,g2,g3,g4;
assign g0 = x2 & x5;
assign g1 = ~1'b0 ^ x3;
assign g2 = x6 & g1;
assign g3 = g0 ^ g2;
assign g4 = x0 & g3;
assign y = g4;
endmodule
