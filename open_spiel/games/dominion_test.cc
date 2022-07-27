#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/dominion.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/algorithm/container.h"

namespace open_spiel {
namespace dominion {
namespace {
using namespace dominion;

namespace testing = open_spiel::testing;

bool IsCopper(const Card* card){
  return card->GetId() == COPPER.GetId();
}
bool IsEstate(const Card* card){
  return card->GetId() == ESTATE.GetId();
}

const GameParameters params =  {
  {"kingdom_cards",GameParameter(kDefaultKingdomCards)}
};

void BasicDominionTests() {
  testing::LoadGameTest("dominion");
}
void InitialDominionGameStateTests(){
  std::shared_ptr<const Game> game = LoadGame("dominion",params);
  DominionState state(game,kDefaultKingdomCards);
  SPIEL_CHECK_EQ(state.getPlayers().size(),2);
  std::vector<std::string> expected_Cards = {"Village", "Laboratory", "Festival", "Market", "Smithy", "Militia", "Gardens", "Chapel", "Witch", "Workshop"};
  SPIEL_DCHECK_EQ(state.GetKingdomCards(),expected_Cards);
}
void InitialPlayerState(){
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion",params);
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
    Action outcome =
        SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    state.DoApplyAction(outcome);
  }
  const int num_coppers_p0 = absl::c_count_if(state.GetPlayerState(0).GetAllCards(),IsCopper);
  const int num_estates_p0 = absl::c_count_if(state.GetPlayerState(0).GetAllCards(),IsEstate);
  const int num_coppers_p1 = absl::c_count_if(state.GetPlayerState(1).GetAllCards(),IsCopper);
  const int num_estates_p1 = absl::c_count_if(state.GetPlayerState(1).GetAllCards(),IsEstate);
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
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
     std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
     bool has_copper = absl::c_find_if(outcomes,[](std::pair<Action,double> outcome){
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
  SPIEL_DCHECK_EQ(state.getPlayers().at(0).GetCoins(),2);  
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetTurnPhase(),BuyPhase);  
}

void BuyTreasureCard() {
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
    std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
    bool has_copper = absl::c_find_if(outcomes,[](std::pair<Action,double> outcome){
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
  std::vector<Action> expected_actions{66,67,70,73,80,82,167};
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetCoins(),3);
  SPIEL_CHECK_EQ(state.LegalActions(),expected_actions);
  state.DoApplyAction(SILVER.GetBuy());
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetCoins(),0);
  SPIEL_CHECK_EQ(state.getPlayers().at(0).GetBuys(),0);
  const int num_silver = absl::c_count_if(state.GetCurrentPlayerState().GetDiscardPile(),[](const Card* card){
    return card->GetId() == SILVER.GetId();
  });
  SPIEL_CHECK_EQ(num_silver,1);
}

void BuyVictoryCard() {
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
    std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
    bool has_copper = absl::c_find_if(outcomes,[](std::pair<Action,double> outcome){
      return outcome.first == COPPER.GetId();
     }) != outcomes.end();
    if(has_copper){
      state.DoApplyAction(COPPER.GetId());
    }else{
      state.DoApplyAction(ESTATE.GetId());
    }
  }
  for (Action action : {COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),END_PHASE_ACTION}){
    state.DoApplyAction(action);
  }
  state.DoApplyAction(DUCHY.GetBuy());
  PlayerState& currentPlayerState = state.GetCurrentPlayerState();
  std::list<const Card*> discard_pile = currentPlayerState.GetDiscardPile();
  const int num_duchy = std::count_if(discard_pile.begin(),discard_pile.end(),[](const Card* card){
    return card->GetId() == DUCHY.GetId();
  });
  SPIEL_CHECK_EQ(num_duchy,1);
  SPIEL_CHECK_EQ(currentPlayerState.GetBuys(),0);
  SPIEL_CHECK_EQ(currentPlayerState.GetCoins(),0);
  SPIEL_CHECK_EQ(currentPlayerState.GetVictoryPoints(),3);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);

}

void SkipBuyPhase() {
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
    std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
    bool has_copper = absl::c_find_if(outcomes,[](std::pair<Action,double> outcome){
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
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
    Action outcome = SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    state.DoApplyAction(outcome);
  }
    
  state.ApplyAction(END_PHASE_ACTION);
  state.ApplyAction(END_PHASE_ACTION);
  state.ApplyAction(END_PHASE_ACTION);
  state.ApplyAction(END_PHASE_ACTION);
  state.ApplyAction(END_PHASE_ACTION);
  state.ApplyAction(END_PHASE_ACTION);

  SPIEL_DCHECK_TRUE(state.GetPlayerState(0).GetDrawPile().empty());
  SPIEL_DCHECK_TRUE(state.GetPlayerState(0).GetHand().empty());
  SPIEL_DCHECK_TRUE(state.GetPlayerState(0).GetAddDiscardPileToDrawPile());
  SPIEL_DCHECK_EQ(state.GetPlayerState(0).GetDiscardPile().size(),10);
  SPIEL_DCHECK_EQ(state.GetPlayerState(0).GetDrawPile().size(),0);

  while(state.IsChanceNode()){
  Action outcome =
      SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
  state.DoApplyAction(outcome);
  }
  SPIEL_DCHECK_EQ(state.GetPlayerState(0).GetDiscardPile().size(),0);
  SPIEL_DCHECK_EQ(state.GetPlayerState(0).GetDrawPile().size(),5);
  SPIEL_DCHECK_EQ(state.GetPlayerState(0).GetHand().size(),5);

}

void TestBuyAction() {
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
    std::vector<std::pair<Action, double>> outcomes = state.ChanceOutcomes();
    bool has_copper = absl::c_find_if(outcomes,[](std::pair<Action,double> outcome){
      return outcome.first == COPPER.GetId();
    }) != outcomes.end();
    if(has_copper){
      state.DoApplyAction(COPPER.GetId());
    }else{
      state.DoApplyAction(ESTATE.GetId());
    }
  }
  for (Action action : {COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),END_PHASE_ACTION}){
    state.DoApplyAction(action);
  }
  state.DoApplyAction(VILLAGE.GetBuy());
  std::list<const Card*> discard_pile = state.GetPlayerState(0).GetDiscardPile();
  const int num_village = std::count_if(discard_pile.begin(),discard_pile.end(),[](const Card* card){
    return card->GetId() == VILLAGE.GetId();
  });
  SPIEL_CHECK_EQ(num_village,1);
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetBuys(),0);
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetCoins(),2);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
}
} // namespace

namespace action_card_tests {
  void PlayVillage() {
    // Playing Village adds 1 card to hand, 2 actions 
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion");
    DominionState state(game,kDefaultKingdomCards);
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      state.DoApplyAction(outcome);
    }
    state.DoApplyAction(END_PHASE_ACTION);
    state.GetCurrentPlayerState().AddFrontToDrawPile(&VILLAGE);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    state.DoApplyAction(END_PHASE_ACTION);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
    state.DoApplyAction(VILLAGE.GetPlay());
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetActions(),2);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetHand().size(),5);
  };
   void TestLaboratory() {
    // Playing Laboratory adds 2 cards, 1 action 
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion");
    DominionState state(game,kDefaultKingdomCards);
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      state.DoApplyAction(outcome);
    }
    state.DoApplyAction(END_PHASE_ACTION);
    state.GetCurrentPlayerState().AddFrontToDrawPile(&LABORATORY);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    state.DoApplyAction(END_PHASE_ACTION);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
    state.DoApplyAction(LABORATORY.GetPlay());
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetDrawPile().size(),1);
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      state.DoApplyAction(outcome);
    }
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetActions(),1);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetHand().size(),6);
  };
  void TestFestival() {
    // Playing Festival adds 2 actions, 1 buys, 2 coins 
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion");
    DominionState state(game,kDefaultKingdomCards);
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      state.DoApplyAction(outcome);
    }
    state.DoApplyAction(END_PHASE_ACTION);
    state.GetCurrentPlayerState().AddFrontToDrawPile(&FESTIVAL);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    state.DoApplyAction(END_PHASE_ACTION);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
    state.DoApplyAction(FESTIVAL.GetPlay());

    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetActions(),2);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetBuys(),2);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetCoins(),2);
  };
  void TestMarket(){
    // Playing Market adds 1 action, 1 buy, 1 coin, 1 card 
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion");
    DominionState state(game,kDefaultKingdomCards);
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      state.DoApplyAction(outcome);
    }
    state.DoApplyAction(END_PHASE_ACTION);
    state.GetCurrentPlayerState().AddFrontToDrawPile(&MARKET);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    state.DoApplyAction(END_PHASE_ACTION);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
    state.DoApplyAction(MARKET.GetPlay());

    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetActions(),1);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetBuys(),2);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetCoins(),1);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetHand().size(),5);
  }
  void TestSmithy(){
    // Playing Smithy adds 3 cards
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion");
    DominionState state(game,kDefaultKingdomCards);
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      state.DoApplyAction(outcome);
    }
    state.DoApplyAction(END_PHASE_ACTION);
    state.GetCurrentPlayerState().AddFrontToDrawPile(&SMITHY);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    state.DoApplyAction(END_PHASE_ACTION);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
    state.DoApplyAction(SMITHY.GetPlay());
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      state.DoApplyAction(outcome);
  }
  SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetHand().size(),7);
} 
void TestMilitia(){
    // // Adds 2 coins, each other player discards down to 3 cards in hand.
    // std::mt19937 rng;
    // std::shared_ptr<const Game> game = LoadGame("dominion");
    // DominionState state(game,kDefaultKingdomCards);
    // while(state.IsChanceNode()){
    //   Action outcome =
    //       SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    //   state.DoApplyAction(outcome);
    // }
    // state.DoApplyAction(END_PHASE_ACTION);
    // state.GetCurrentPlayerState().AddFrontToDrawPile(&MILITIA);
    // state.DoApplyAction(END_PHASE_ACTION);
    // SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    // state.DoApplyAction(END_PHASE_ACTION);
    // state.DoApplyAction(END_PHASE_ACTION);
    // SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    // SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
    // state.DoApplyAction(MILITIA.GetPlay());
    // SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetCoins(),2);
} 
void TestGardens(){
    // Worth 1 victory point per 10 cards you have (round down)
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion");
    DominionState state(game,kDefaultKingdomCards);
    while(state.IsChanceNode()){
      Action outcome =
          SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
      state.DoApplyAction(outcome);
    }
    state.DoApplyAction(END_PHASE_ACTION);
    state.GetCurrentPlayerState().AddFrontToDrawPile(&GARDENS);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    state.DoApplyAction(END_PHASE_ACTION);
    state.DoApplyAction(END_PHASE_ACTION);
    SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    SPIEL_DCHECK_EQ(state.GetCurrentPlayerState().GetVictoryPoints(),4);

} 
// void TestChapel(){
//   // trash up to four cards in your hand.
//   std::mt19937 rng;
//   std::shared_ptr<const Game> game = LoadGame("dominion");
//   DominionState state(game,kDefaultKingdomCards);
//   while(state.IsChanceNode()){
//     Action outcome =
//         SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
//     state.DoApplyAction(outcome);
//   }
//   state.DoApplyAction(END_PHASE_ACTION);
//   state.GetCurrentPlayerState().AddFrontToDrawPile(&MILITIA);
//   state.DoApplyAction(END_PHASE_ACTION);
//   SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
//   state.DoApplyAction(END_PHASE_ACTION);
//   state.DoApplyAction(END_PHASE_ACTION);
//   SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
//   SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
//   state.DoApplyAction(MILITIA.GetPlay());
//   SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetCoins(),2);
// } 
void TestWitch(){
    // Adds 2 coins, each other player discards down to 3 cards in hand.
    // std::mt19937 rng;
    // std::shared_ptr<const Game> game = LoadGame("dominion");
    // DominionState state(game,kDefaultKingdomCards);
    // while(state.IsChanceNode()){
    //   Action outcome =
    //       SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    //   state.DoApplyAction(outcome);
    // }
    // state.DoApplyAction(END_PHASE_ACTION);
    // state.GetCurrentPlayerState().AddFrontToDrawPile(&MILITIA);
    // state.DoApplyAction(END_PHASE_ACTION);
    // SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    // state.DoApplyAction(END_PHASE_ACTION);
    // state.DoApplyAction(END_PHASE_ACTION);
    // SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    // SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
    // state.DoApplyAction(MILITIA.GetPlay());
    // SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetCoins(),2);
} 
void TestWorkshop(){
    // Adds 2 coins, each other player discards down to 3 cards in hand.
    // std::mt19937 rng;
    // std::shared_ptr<const Game> game = LoadGame("dominion");
    // DominionState state(game,kDefaultKingdomCards);
    // while(state.IsChanceNode()){
    //   Action outcome =
    //       SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    //   state.DoApplyAction(outcome);
    // }
    // state.DoApplyAction(END_PHASE_ACTION);
    // state.GetCurrentPlayerState().AddFrontToDrawPile(&MILITIA);
    // state.DoApplyAction(END_PHASE_ACTION);
    // SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
    // state.DoApplyAction(END_PHASE_ACTION);
    // state.DoApplyAction(END_PHASE_ACTION);
    // SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
    // SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
    // state.DoApplyAction(MILITIA.GetPlay());
    // SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetCoins(),2);
} 

void TestCellar(){
   //Discard any number of cards, then draw that many.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Cellar");
  while(state.IsChanceNode()){
    Action outcome =
        SampleAction(state.ChanceOutcomes(),std::uniform_real_distribution<double>(0.0, 1.0)(rng)).first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&CELLAR);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(CELLAR.GetPlay());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);

  std::set<Action> moves;
  for(const Card* card: state.GetCurrentPlayerState().GetHand()){
    moves.insert(card->GetDiscard());
  }
  moves.insert(END_PHASE_ACTION);
  std::vector<Action> legal_actions(moves.begin(),moves.end());
  absl::c_sort(legal_actions);
  SPIEL_CHECK_EQ(state.LegalActions(),legal_actions);

  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_DCHECK_EQ(state.GetCurrentPlayerState().GetHand().size(),kHandSize-1);

}

}//namespace action_card_tests
}  // namespace dominion
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::dominion::BasicDominionTests();
  open_spiel::dominion::InitialDominionGameStateTests();
  open_spiel::dominion::InitialPlayerState();
  open_spiel::dominion::PlayTreasureCard();
  open_spiel::dominion::BuyTreasureCard();
  open_spiel::dominion::BuyVictoryCard();
  open_spiel::dominion::SkipBuyPhase();
  open_spiel::dominion::TestBuyAction();
  open_spiel::dominion::TestEndTurnAddCardsFromDisacrdToDrawPile();
  // open_spiel::dominion::action_card_tests::PlayVillage();
  open_spiel::dominion::action_card_tests::TestLaboratory();
  open_spiel::dominion::action_card_tests::TestFestival();
  open_spiel::dominion::action_card_tests::TestMarket();
  open_spiel::dominion::action_card_tests::TestSmithy();
  open_spiel::dominion::action_card_tests::TestMilitia();
  open_spiel::dominion::action_card_tests::TestGardens();
  open_spiel::dominion::action_card_tests::TestCellar();
  // open_spiel::dominion::action_card_tests::TestWitch();
  // open_spiel::dominion::action_card_tests::TestWorkshop();

}
