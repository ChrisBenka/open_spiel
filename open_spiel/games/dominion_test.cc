#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/dominion.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace domninion {
namespace game_setup {

namespace testing = open_spiel::testing;
using namespace dominion;

bool IsCopper(Card card){
  return card.GetId() == COPPER.GetId();
}
bool IsEstate(Card card){
  return card.GetId() == ESTATE.GetId();
}

void BasicDominionTests() {
  testing::LoadGameTest("dominion");
}
void InitialDominionGameStateTests(){
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  SPIEL_CHECK_EQ(state.getPlayers().size(),2);
}
void EachPlayerStartsWith7Coppers3Estates(){
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion");
    DominionState state(game);
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
      state.DoApplyAction(outcome);
    }
    std::vector<Card> player_0_draw_pile = state.getPlayers().at(0).GetDrawPile();
    std::vector<Card> player_1_draw_pile = state.getPlayers().at(1).GetDrawPile();
    const int num_coppers_p0 = std::count_if(player_0_draw_pile.begin(),player_0_draw_pile.end(),IsCopper);
    const int num_estates_p0 = std::count_if(player_0_draw_pile.begin(),player_0_draw_pile.end(),IsEstate);
    const int num_coppers_p1 = std::count_if(player_1_draw_pile.begin(),player_1_draw_pile.end(),IsCopper);
    const int num_estates_p1 = std::count_if(player_1_draw_pile.begin(),player_1_draw_pile.end(),IsEstate);
    SPIEL_CHECK_EQ(num_coppers_p0,kInitCoppers);
    SPIEL_CHECK_EQ(num_estates_p0,kInitEstates);
    SPIEL_CHECK_EQ(num_coppers_p1,kInitCoppers);
    SPIEL_CHECK_EQ(num_estates_p1,kInitEstates);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
} 

void isTerminalTests(){
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  SPIEL_CHECK_FALSE(state.IsTerminal());
}



}  // namespace
}  // namespace dominion
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::domninion::game_setup::BasicDominionTests();
  open_spiel::domninion::game_setup::InitialDominionGameStateTests();
  open_spiel::domninion::game_setup::EachPlayerStartsWith7Coppers3Estates();
}
