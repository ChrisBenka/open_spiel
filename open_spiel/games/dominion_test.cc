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
          SampleAction(state.ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
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
  std::vector<Action> expected_actions{END_PHASE_ACTION,COPPER.GetPlay()};
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
  std::vector<Action> expected_actions{0,67,68,71,74,81,83};
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
  SPIEL_CHECK_EQ(currentPlayerState.GetVictoryPoints(),6);
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
    Action outcome = state.ChanceOutcomes().front().first;
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
  Action outcome = state.ChanceOutcomes().front().first;
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
      Action outcome = state.ChanceOutcomes().front().first;
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
      Action outcome = state.ChanceOutcomes().front().first;
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
      Action outcome = state.ChanceOutcomes().front().first;
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
      Action outcome = state.ChanceOutcomes().front().first;
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
      Action outcome = state.ChanceOutcomes().front().first;
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
      Action outcome = state.ChanceOutcomes().front().first;
      state.DoApplyAction(outcome);
  }
  SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetHand().size(),7);
} 
void TestMilitiaOpponentHasNoMoat(){
  // add 2 coins each other players discards down to 3 cards
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Cellar");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MILITIA);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(MILITIA.GetPlay());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  while(state.GetEffectRunner()->Active()){
    state.DoApplyAction(state.LegalActions().front());
  }
  SPIEL_CHECK_EQ(state.GetPlayerState(1).GetHand().size(),3);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
} 
void TestMilitiaOpponentRevealsMoat(){
  // add 2 coins each other players discards down to 3 cards
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Moat");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MILITIA);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MOAT);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(MILITIA.GetPlay());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  std::vector<Action> moves = {END_PHASE_ACTION,MOAT.GetPlay()};
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(MOAT.GetPlay());
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
} 
void TestMilitiaOpponentChoosesNotToRevealMoat(){
  // add 2 coins each other players discards down to 3 cards
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Moat");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MILITIA);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MOAT);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  state.DoApplyAction(MILITIA.GetPlay());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  std::vector<Action> moves = {END_PHASE_ACTION,MOAT.GetPlay()};
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  while(state.GetEffectRunner()->Active()){
    state.DoApplyAction(state.LegalActions().front());
  }
  SPIEL_CHECK_EQ(state.GetPlayerState(1).GetHand().size(),3);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
} 
void TestWitchOpponentHasNoMoat(){
  // Opponents gain a curse card. Player gains 2 cards to hand.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Moat");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&WITCH);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  state.DoApplyAction(WITCH.GetPlay());

  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }

  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetHand().size(),6);
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(absl::c_count(state.GetPlayerState(1).GetAllCards(),&CURSE),1);
  SPIEL_CHECK_EQ(state.GetPlayerState(1).GetVictoryPoints(),2);
} 

void TestWitchOpponentRevealsMoat(){
   // Opponents is immune, does not gain a curse card. Player gains 2 cards to hand.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Moat");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&WITCH);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MOAT);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  state.DoApplyAction(WITCH.GetPlay());

  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetHand().size(),6);
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  state.DoApplyAction(MOAT.GetPlay());
  SPIEL_CHECK_EQ(absl::c_count(state.GetPlayerState(1).GetAllCards(),&CURSE),0);
} 

void TestWitchOpponentChoosesNotToRevealMoat(){
    // Opponents is immune, does not gain a curse card. Player gains 2 cards to hand.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Moat");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&WITCH);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MOAT);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  state.DoApplyAction(WITCH.GetPlay());

  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetHand().size(),6);
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(absl::c_count(state.GetPlayerState(1).GetAllCards(),&CURSE),1);
  SPIEL_CHECK_EQ(state.GetPlayerState(1).GetVictoryPoints(),2);
} 

void TestGardens(){
    // Worth 1 victory point per 10 cards you have (round down)
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion");
    DominionState state(game,kDefaultKingdomCards);
    while(state.IsChanceNode()){
      Action outcome = state.ChanceOutcomes().front().first;
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
void TestChapel(){
  // trash up to four cards in your hand.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
       Action outcome =
          SampleAction(state.ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&CHAPEL);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
  state.DoApplyAction(CHAPEL.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  state.DoApplyAction(COPPER.GetTrash());
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetTrashPile().size(),1);  
} 
void TestChapelTrashFourCards(){
  // trash up to four cards in your hand.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,kDefaultKingdomCards);
  while(state.IsChanceNode()){
       Action outcome =
          SampleAction(state.ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&CHAPEL);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetTurnPhase(),ActionPhase);
  state.DoApplyAction(CHAPEL.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  for(int i = 0; i < 4; i++){
    Action action = state.LegalActions().front();
    state.DoApplyAction(action);
  }
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetTrashPile().size(),4);  
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetHand().size(),0);  
}
void TestWorkshop(){
    // Gain a card costing up to 4 coins
    // Opponents is immune, does not gain a curse card. Player gains 2 cards to hand.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Workshop;Moat");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&WORKSHOP);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  state.DoApplyAction(WORKSHOP.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  state.DoApplyAction(VILLAGE.GetGain());
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDiscardPile(),[](const Card* c){
    return c->GetId() == VILLAGE.GetId();
  }),1);
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
} 

void TestCellar(){
   //Discard any number of cards, then draw that many.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Witch;Cellar");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
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

  state.DoApplyAction(COPPER.GetDiscard());
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_FALSE(state.IsChanceNode());
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetHand().size(),kHandSize-1);
}

void TestBanditNoCardsToTrash() {
  //Player will gain a gold, opponents will discard top two cards
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Bandit;Cellar");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&BANDIT);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(BANDIT.GetPlay());
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDiscardPile(),[](const Card* c){
    return c->GetId() == GOLD.GetId();
  }),1);
  SPIEL_CHECK_TRUE(state.IsChanceNode());
  state.getPlayers().at(1).AddFrontToDrawPile(&COPPER);
  state.getPlayers().at(1).AddFrontToDrawPile(&COPPER);
  while(state.IsChanceNode()){
       Action outcome =
          SampleAction(state.ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
    state.DoApplyAction(outcome);
  }
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  std::vector<Action> moves = {COPPER.GetDiscard()};
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(COPPER.GetDiscard());
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(COPPER.GetDiscard());
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
}

void TestBanditHasTwoCardsToSelectToTrash() {
  //Player will gain a gold, opponents will trash one of top two cards and discard the other.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Bandit;Cellar");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&BANDIT);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(BANDIT.GetPlay());
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDiscardPile(),[](const Card* c){
    return c->GetId() == GOLD.GetId();
  }),1);
  SPIEL_CHECK_TRUE(state.IsChanceNode());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  state.getPlayers().at(1).AddFrontToDrawPile(&SILVER);
  state.getPlayers().at(1).AddFrontToDrawPile(&GOLD);
  while(state.IsChanceNode()){
       Action outcome =
          SampleAction(state.ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
    state.DoApplyAction(outcome);
  }
  std::vector<Action> moves = {SILVER.GetTrash(),GOLD.GetTrash()};
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(GOLD.GetTrash());
  moves = {SILVER.GetDiscard()};
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(SILVER.GetDiscard());
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
}

void TestRemodel() {
  //Player will gain a gold, opponents will trash one of top two cards and discard the other.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Bandit;Cellar");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&SILVER);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&REMODEL);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(REMODEL.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  state.DoApplyAction(SILVER.GetTrash());;
  std::vector<Action> moves;
  for(auto pile_pair : state.getSupplyPiles()){
    if(!pile_pair.second.Empty() && pile_pair.second.getCard()->GetCost() <= SILVER.GetCost() + 2){
      moves.push_back(pile_pair.second.getCard()->GetGain());
    }
  }
  absl::c_sort(moves);
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(VILLAGE.GetGain());
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDiscardPile(),[](const Card* c){
    return c->GetId() == VILLAGE.GetId();
  }),1);

}
void TestMoneyLenderTrashCopper(){
    //Player will gain a gold, opponents will trash one of top two cards and discard the other.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Cellar");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MONEYLENDER);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(MONEYLENDER.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  std::vector<Action> moves = {COPPER.GetTrash(),END_PHASE_ACTION};
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(COPPER.GetTrash());
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetCoins(),3);
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetTrashPile(),[](const Card* c){
    return c->GetId() == COPPER.GetId();
  }),1);
}
void TestMoneyLenderDoNotTrashCopper(){
    //Player will gain a gold, opponents will trash one of top two cards and discard the other.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Cellar");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MONEYLENDER);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(MONEYLENDER.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  std::vector<Action> moves = {COPPER.GetTrash(),END_PHASE_ACTION};
  SPIEL_CHECK_EQ(state.LegalActions(),moves);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetCoins(),0);
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetTrashPile(),[](const Card* c){
    return c->GetId() == COPPER.GetId();
  }),0);
}

void testPoacherNoEmptyPiles(){
  //Discard a card per empty supply pile
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Poacher");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&POACHER);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(POACHER.GetPlay());
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
}

void testPoacherEmptyPiles(){
  //Discard a card per empty supply pile
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Poacher");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&POACHER);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.getSupplyPiles().at("Militia").Clear();
  state.getSupplyPiles().at("Gardens").Clear();
  state.getSupplyPiles().at("Copper").Clear();
  state.DoApplyAction(POACHER.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  for(int i = 0; i < 3; i++){
    state.DoApplyAction(state.LegalActions().front());
  }  
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetHand().size(),2);
}

void TestMineDoNotTrash(){
  //Trash a Treasure card from your hand. Gain a Treasure card costing up to 3 coins more; put it into your hand.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Mine");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MINE);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(MINE.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  std::vector<Action> legal_actions = {END_PHASE_ACTION,COPPER.GetTrash()};
  SPIEL_CHECK_EQ(state.LegalActions(),legal_actions);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
}
void TestMineTrashTreasureCard(){
  //Trash a Treasure card from your hand. Gain a Treasure card costing up to 3 coins more; put it into your hand.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Mine");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MINE);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(MINE.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  std::vector<Action> legal_actions = {END_PHASE_ACTION,COPPER.GetTrash()};
  SPIEL_CHECK_EQ(state.LegalActions(),legal_actions);
  state.DoApplyAction(COPPER.GetTrash());
  legal_actions = {COPPER.GetGain(),SILVER.GetGain()};     
  state.DoApplyAction(SILVER.GetGain());
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetTrashPile(),[](const Card* c){
    return c->GetId() == COPPER.GetId();
  }),1);
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDiscardPile(),[](const Card* c){
    return c->GetId() == SILVER.GetId();
  }),1);
}

void TestVassalTopCardIsNotActionCard(){
  //Discard the top card of your deck.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Vassal");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&VASSAL);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(VASSAL.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  state.GetCurrentPlayerState().AddFrontToDrawPile(&SILVER);
  std::vector<Action> legal_actions = {SILVER.GetDiscard()};
  SPIEL_CHECK_EQ(state.LegalActions(),legal_actions);
  state.DoApplyAction(SILVER.GetDiscard());
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDiscardPile(),[](const Card* c){
    return c->GetId() == SILVER.GetId();
  }),1);
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDrawPile(),[](const Card* c){
    return c->GetId() == SILVER.GetId();
  }),0);
}

void TestVassalTopCardIsActionCard(){
  //Play or discard top of your deck
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Vassal");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&VASSAL);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(VASSAL.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  state.GetCurrentPlayerState().AddFrontToDrawPile(&MILITIA);
  std::vector<Action> legal_actions = {MILITIA.GetPlay(),MILITIA.GetDiscard()};
  SPIEL_CHECK_EQ(state.LegalActions(),legal_actions);
  state.DoApplyAction(MILITIA.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  // opponents need to discard down to 3 cards
  for(int i = 0; i < 3; i++){
    state.DoApplyAction(state.LegalActions().front());
  }
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetCoins(),2);

  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDiscardPile(),[](const Card* c){
    return c->GetId() == MILITIA.GetId();
  }),0);
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetDrawPile(),[](const Card* c){
    return c->GetId() == MILITIA.GetId();
  }),0);
}

void TestCouncilRoom(){
  //+4 Cards, +1 Buy, Each other play draws a card.
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Council Room");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&COUNCIL_ROOM);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(COUNCIL_ROOM.GetPlay());
  SPIEL_CHECK_TRUE(state.IsChanceNode());
  while(state.IsChanceNode()){
       Action outcome =
          SampleAction(state.ChanceOutcomes(),
                       std::uniform_real_distribution<double>(0.0, 1.0)(rng))
              .first;
    state.DoApplyAction(outcome);
  }
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
  SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetBuys(),2);
  SPIEL_CHECK_EQ(state.GetCurrentPlayerState().GetHand().size(),8);
  SPIEL_CHECK_EQ(state.GetPlayerState(1).GetHand().size(),6);
}

void TestArtisan(){
  // Gain a card to your hand costing up to 5 coins
  std::mt19937 rng;
  std::shared_ptr<const Game> game = LoadGame("dominion");
  DominionState state(game,"Village;Laboratory;Festival;Market;Smithy;Militia;Gardens;Chapel;Moneylender;Artisan");
  while(state.IsChanceNode()){
    Action outcome = state.ChanceOutcomes().front().first;
    state.DoApplyAction(outcome);
  }
  state.DoApplyAction(END_PHASE_ACTION);
  state.GetCurrentPlayerState().AddFrontToDrawPile(&ARTISAN);
  state.DoApplyAction(END_PHASE_ACTION);
  SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
  state.DoApplyAction(END_PHASE_ACTION);
  state.DoApplyAction(END_PHASE_ACTION);

  state.DoApplyAction(ARTISAN.GetPlay());
  SPIEL_CHECK_TRUE(state.GetEffectRunner()->Active());
  state.DoApplyAction(MILITIA.GetGain());
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetHand(),[](const Card* c){
    return c->GetId() == MILITIA.GetId();
  }),1);
  state.DoApplyAction(MILITIA.GetPlaceOntoDeck());
  SPIEL_CHECK_EQ(absl::c_count_if(state.GetPlayerState(0).GetHand(),[](const Card* c){
    return c->GetId() == MILITIA.GetId();
  }),0);
  SPIEL_CHECK_EQ(state.GetPlayerState(0).GetDrawPile().front()->GetId(),MILITIA.GetId());
  SPIEL_CHECK_FALSE(state.GetEffectRunner()->Active());
}


}//namespace action_card_tests

namespace observer {
  void ObservationTensor(){
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion",params);
    std::unique_ptr<State> state = game->NewInitialState();
    while(state->IsChanceNode()){
      Action outcome = state->ChanceOutcomes().front().first;
      state->ApplyAction(outcome);
    }
    std::vector<float> obs = state->ObservationTensor();
    std::vector<float> expected = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      46, 40, 30, // Treasure Supply
      10, 8, 8, 8, // Victory Supply
      10, 10, 10, 10, 10, 10, 8, 10, 10, 10, // Kingdom Supply
      1 ,1, 1, 0,  // TurnPhase + Player ABC
      -1, //Effect
      1, 1, 1, 1, 1, //hand
      2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // DRAW
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // DISCARD
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // TRASH
      7, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // Opponents Playable cards
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 //  Opponents Trashed cards
    };
    // 
    SPIEL_CHECK_EQ(obs,expected);
  }
  void ObservationString(){
    std::mt19937 rng;
    std::shared_ptr<const Game> game = LoadGame("dominion",params);
    std::unique_ptr<State> state = game->NewInitialState();
    while(state->IsChanceNode()){
      Action outcome = state->ChanceOutcomes().front().first;
      state->ApplyAction(outcome);
    }
    std::string obs_str = state->ObservationString();
    SPIEL_CHECK_NE("",obs_str);
  }
}


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
  open_spiel::dominion::action_card_tests::PlayVillage();
  open_spiel::dominion::action_card_tests::TestLaboratory();
  open_spiel::dominion::action_card_tests::TestFestival();
  open_spiel::dominion::action_card_tests::TestMarket();
  // // open_spiel::dominion::action_card_tests::TestSmithy();
  open_spiel::dominion::action_card_tests::TestMilitiaOpponentHasNoMoat();
  open_spiel::dominion::action_card_tests::TestMilitiaOpponentRevealsMoat();
  open_spiel::dominion::action_card_tests::TestMilitiaOpponentChoosesNotToRevealMoat();
  open_spiel::dominion::action_card_tests::TestGardens();
  open_spiel::dominion::action_card_tests::TestCellar();
  // open_spiel::dominion::action_card_tests::TestChapelTrashFourCards();
  open_spiel::dominion::action_card_tests::TestWitchOpponentHasNoMoat();
  open_spiel::dominion::action_card_tests::TestWitchOpponentRevealsMoat();
  open_spiel::dominion::action_card_tests::TestWitchOpponentChoosesNotToRevealMoat();
  open_spiel::dominion::action_card_tests::TestWorkshop();
  // // open_spiel::dominion::action_card_tests::TestBanditNoCardsToTrash();
  open_spiel::dominion::action_card_tests::TestBanditHasTwoCardsToSelectToTrash();
  open_spiel::dominion::action_card_tests::TestRemodel();
  open_spiel::dominion::action_card_tests::TestMoneyLenderTrashCopper();
  open_spiel::dominion::action_card_tests::TestMoneyLenderDoNotTrashCopper();
  open_spiel::dominion::action_card_tests::testPoacherNoEmptyPiles();
  open_spiel::dominion::action_card_tests::testPoacherEmptyPiles();
  open_spiel::dominion::action_card_tests::TestMineDoNotTrash();
  open_spiel::dominion::action_card_tests::TestMineTrashTreasureCard();
  open_spiel::dominion::action_card_tests::TestVassalTopCardIsNotActionCard();
  open_spiel::dominion::action_card_tests::TestVassalTopCardIsActionCard();
  open_spiel::dominion::action_card_tests::TestCouncilRoom();
  open_spiel::dominion::action_card_tests::TestArtisan();
  open_spiel::dominion::observer::ObservationTensor();
  open_spiel::dominion::observer::ObservationString();
}
