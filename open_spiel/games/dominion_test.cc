#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/dominion.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace domninion {
using namespace dominion;
namespace {

namespace testing = open_spiel::testing;

bool IsCopper(const Card* card){
  return card->GetId() == COPPER.GetId();
}
bool IsEstate(const Card* card){
  return card->GetId() == ESTATE.GetId();
}

void BasicDominionTests() {
  testing::LoadGameTest("dominion");
}
void InitialDominionGameStateTests(){
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  SPIEL_CHECK_EQ(state.getPlayers().size(),2);
}
void InitialPlayerState(){
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  while(state.IsChanceNode()){
    Action outcome =
        SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    state.DoApplyAction(outcome);
  }
  std::list<const Card*> player_0_all_cards = state.getPlayers().at(0).GetAllCards();
  std::list<const Card*> player_1_all_cards = state.getPlayers().at(1).GetAllCards();
  const int num_coppers_p0 = std::count_if(player_0_all_cards.begin(),player_0_all_cards.end(),IsCopper);
  const int num_estates_p0 = std::count_if(player_0_all_cards.begin(),player_0_all_cards.end(),IsEstate);
  const int num_coppers_p1 = std::count_if(player_1_all_cards.begin(),player_1_all_cards.end(),IsCopper);
  const int num_estates_p1 = std::count_if(player_1_all_cards.begin(),player_1_all_cards.end(),IsEstate);
  SPIEL_CHECK_EQ(player_0_all_cards.size(),kInitSupply);
  SPIEL_CHECK_EQ(player_1_all_cards.size(),kInitSupply);
  SPIEL_CHECK_EQ(num_coppers_p0,kInitCoppers);
  SPIEL_CHECK_EQ(num_estates_p0,kInitEstates);
  SPIEL_CHECK_EQ(num_coppers_p1,kInitCoppers);
  SPIEL_CHECK_EQ(num_estates_p1,kInitEstates);
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetHand().size(),kHandSize);
  SPIEL_CHECK_EQ(state.getPlayers().at(1).GetHand().size(),kHandSize);
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetActions(),1);
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetBuys(),1);
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetCoins(),0);
}

void PlayTreasureCard() {
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  while(state.IsChanceNode()){
     std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
     bool has_copper = std::find_if(outcomes.begin(),outcomes.end(),[](std::pair<Action,double> outcome){
      return outcome.first == COPPER.GetId();
     }) != outcomes.end();
    if(has_copper){
      state.DoApplyAction(COPPER.GetId());
    }else{
      state.DoApplyAction(ESTATE.GetId());
    }
  }
  std::vector<Action> expected_actions{COPPER.GetPlay(),END_PHASE_ACTION};
  SPIEL_CHECK_EQ(state.LegalActions(),expected_actions);
  for (Action action : {COPPER.GetPlay(),COPPER.GetPlay(),END_PHASE_ACTION}){
    state.DoApplyAction(action);
  }
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetCoins(),2);  
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetTurnPhase(),BuyPhase);  
}

void BuyTreasureCard() {
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  while(state.IsChanceNode()){
    std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
    bool has_copper = std::find_if(outcomes.begin(),outcomes.end(),[](std::pair<Action,double> outcome){
      return outcome.first == COPPER.GetId();
     }) != outcomes.end();
    if(has_copper){
      state.DoApplyAction(COPPER.GetId());
    }else{
      state.DoApplyAction(ESTATE.GetId());
    }
  }
  for (Action action : {COPPER.GetPlay(),COPPER.GetPlay(),END_PHASE_ACTION}){
    state.DoApplyAction(action);
  }
  std::set<Action> cards_costing_less_eq_2;
  for(const Card* card : all_cards){
    if(card->GetCost() <=2){
      cards_costing_less_eq_2.insert(card->GetBuy());
    }
  }
  std::vector<Action> legal_actions(cards_costing_less_eq_2.begin(),cards_costing_less_eq_2.end());
  legal_actions.push_back(END_PHASE_ACTION);
  std::sort(legal_actions.begin(),legal_actions.end());
  SPIEL_CHECK_EQ(state.LegalActions(),legal_actions);
  state.DoApplyAction(SILVER.GetBuy());
  PlayerState currentPlayerState = state.getPlayers().at(0);
  std::list<const Card*> draw_pile = currentPlayerState.GetDrawPile();
  const int num_silver = std::count_if(draw_pile.begin(),draw_pile.end(),[](const Card* card){
    return card->GetId() == SILVER.GetId();
  });
  SPIEL_CHECK_EQ(num_silver,1);
  SPIEL_CHECK_EQ(currentPlayerState.GetBuys(),0);
  SPIEL_CHECK_EQ(currentPlayerState.GetCoins(),0);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
}

void BuyVictoryCard() {
 std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  while(state.IsChanceNode()){
    std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
    bool has_copper = std::find_if(outcomes.begin(),outcomes.end(),[](std::pair<Action,double> outcome){
      return outcome.first == COPPER.GetId();
     }) != outcomes.end();
    if(has_copper){
      state.DoApplyAction(COPPER.GetId());
    }else{
      state.DoApplyAction(ESTATE.GetId());
    }
  }
  for (Action action : {COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),END_PHASE_ACTION}){
    state.DoApplyAction(action);
  }
  state.DoApplyAction(DUCHY.GetBuy());
  PlayerState currentPlayerState = state.getPlayers().at(0);
  std::list<const Card*> draw_pile = currentPlayerState.GetDrawPile();
  const int num_duchy = std::count_if(draw_pile.begin(),draw_pile.end(),[](const Card* card){
    return card->GetId() == DUCHY.GetId();
  });
  SPIEL_CHECK_EQ(num_duchy,1);
  SPIEL_CHECK_EQ(currentPlayerState.GetBuys(),0);
  SPIEL_CHECK_EQ(currentPlayerState.GetCoins(),0);
  SPIEL_CHECK_EQ(currentPlayerState.GetVictoryPoints(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
}
void SkipBuyPhase() {
 std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  while(state.IsChanceNode()){
    std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
    bool has_copper = std::find_if(outcomes.begin(),outcomes.end(),[](std::pair<Action,double> outcome){
      return outcome.first == COPPER.GetId();
     }) != outcomes.end();
    if(has_copper){
      state.DoApplyAction(COPPER.GetId());
    }else{
      state.DoApplyAction(ESTATE.GetId());
    }
  }
  for (Action action : {COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),END_PHASE_ACTION}){
    state.DoApplyAction(action);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
}

void TestEndTurnAddCardsFromDisacrdToDrawPile(){
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game);
  while(state.IsChanceNode()){
    Action outcome = SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    state.DoApplyAction(outcome);
  }
  for(int i = 0; i < 6; i++){
    state.DoApplyAction(END_PHASE_ACTION);
  }

  SPIEL_CHECK_TRUE(state.GetPlayerState(0).GetDrawPile().empty());
  SPIEL_CHECK_TRUE(state.GetPlayerState(0).GetHand().empty());
  SPIEL_CHECK_TRUE(state.GetPlayerState(0).GetAddDiscardPileToDrawPile());
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetDiscardPile().size(),10);
  while(state.IsChanceNode()){
    Action outcome =
        SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    state.DoApplyAction(outcome);
  }
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetDiscardPile().size(),0);
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetDrawPile().size(),5);
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetHand().size(),5);

}



} // namespace
}  // namespace dominion
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::domninion::BasicDominionTests();
  open_spiel::domninion::InitialDominionGameStateTests();
  open_spiel::domninion::InitialPlayerState();
  open_spiel::domninion::PlayTreasureCard();
  open_spiel::domninion::BuyTreasureCard();
  open_spiel::domninion::SkipBuyPhase();
  open_spiel::domninion::TestEndTurnAddCardsFromDisacrdToDrawPile();
}
