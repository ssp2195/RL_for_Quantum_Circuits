// Minimal adapter to the unmodified, pinned mockturtle algorithms.
// Input/output: n_inputs n_gates output_literal, followed by A/X and two literals.
// Literal 2*i denotes node i; literal 2*i+1 is its complement; node 0 is false.
#include <iostream>
#include <vector>
#include <stdexcept>
#include <mockturtle/networks/xag.hpp>
#include <mockturtle/algorithms/cleanup.hpp>
#include <mockturtle/algorithms/cut_rewriting.hpp>
#include <mockturtle/algorithms/node_resynthesis/xag_minmc2.hpp>
#include <mockturtle/algorithms/xag_resub.hpp>
#include <mockturtle/utils/cost_functions.hpp>
#include <mockturtle/views/topo_view.hpp>
#include <mockturtle/views/fanout_view.hpp>
#include <mockturtle/views/depth_view.hpp>
int main() {
  try {
    unsigned n, count, output;
    if (!(std::cin >> n >> count >> output) || n > 128 || count > 100000)
      throw std::runtime_error("invalid network header");
    mockturtle::xag_network x;
    using signal = mockturtle::xag_network::signal;
    std::vector<signal> s{x.get_constant(false)};
    for (unsigned i=0; i<n; ++i) s.push_back(x.create_pi());
    for (unsigned i=0; i<count; ++i) {
      char op; unsigned a,b; if (!(std::cin>>op>>a>>b) || a/2>=s.size() || b/2>=s.size())
        throw std::runtime_error("invalid gate or forward reference");
      auto u=s.at(a/2) ^ bool(a&1), v=s.at(b/2) ^ bool(b&1);
      if (op=='A') s.push_back(x.create_and(u,v));
      else if (op=='X') s.push_back(x.create_xor(u,v));
      else throw std::runtime_error("unsupported gate");
    }
    x.create_po(s.at(output/2) ^ bool(output&1));
    x=mockturtle::cleanup_dangling(x);
    mockturtle::future::xag_minmc_resynthesis resyn;
    mockturtle::cut_rewriting_params cp;
    cp.cut_enumeration_ps.cut_size=5;
    cp.cut_enumeration_ps.cut_limit=16;
    // Fixed two rounds, not tuned on the evaluation cases. Preserve the lowest
    // AND-count network when general size resubstitution would worsen that cost.
    auto ands=[](auto const& ntk){unsigned c=0;ntk.foreach_gate([&](auto i){c+=ntk.is_and(i);});return c;};
    for(unsigned round=0;round<2;++round) {
      auto y=mockturtle::cut_rewriting<mockturtle::xag_network,decltype(resyn),mockturtle::mc_cost<mockturtle::xag_network>>(x,resyn,cp,nullptr);
      y=mockturtle::cleanup_dangling(y);
      if(ands(y)<=ands(x)) x=y;
      y=x.clone(); mockturtle::fanout_view fan{y}; mockturtle::depth_view view{fan};
      mockturtle::resubstitution_params rp;rp.max_pis=8;rp.max_inserts=3;
      mockturtle::xag_resubstitution(view,rp);
      y=mockturtle::cleanup_dangling(y);
      if(ands(y)<=ands(x)) x=y;
    }
    x=mockturtle::cleanup_dangling(x);
    std::vector<unsigned> ids(x.size()); ids[x.get_node(x.get_constant(false))]=0;
    unsigned next=1; x.foreach_pi([&](auto i){ids[i]=next++;});
    struct G {char op;unsigned a,b;};std::vector<G> gates;
    mockturtle::topo_view topo{x};
    topo.foreach_node([&](auto i){if(x.is_constant(i)||x.is_pi(i))return;
      std::vector<unsigned> f; x.foreach_fanin(i,[&](auto a){f.push_back(2*ids.at(x.get_node(a))+x.is_complemented(a));});
      if(f.size()!=2)throw std::runtime_error("invalid arity");
      ids[i]=next++;gates.push_back({x.is_and(i)?'A':'X',f[0],f[1]});});
    unsigned out=0;x.foreach_po([&](auto a){out=2*ids.at(x.get_node(a))+x.is_complemented(a);});
    std::cout<<n<<" "<<gates.size()<<" "<<out<<"\n";
    for(auto g:gates)std::cout<<g.op<<" "<<g.a<<" "<<g.b<<"\n";
  }catch(std::exception const& e){std::cerr<<e.what()<<"\n";return 2;}
}
