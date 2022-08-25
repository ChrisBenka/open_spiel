#include <algorithm>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel_bots.h"


#include "open_spiel/bots/dominion/big_money_bot.h"
#include "open_spiel/games/dominion.h"

namespace open_spiel {
 namespace dominion {
  ActionsAndProbs BigMoneyBot::GetPolicy(const State& state) {
   std::vector<Action> legal_actions = state.LegalActions(player_id_);
   ActionsAndProbs policy;
   if(legal_actions.size() == 1 && legal_actions.front() == END_PHASE_ACTION){
    policy = {std::make_pair(END_PHASE_ACTION,1)};
   }else{

   }
   return policy;
  }

  std::pair<ActionsAndProbs, Action> BigMoneyBot::StepWithPolicy(const State& state)  {
   return {};
  }
  DominionObservation BigMoneyBot::GetObservation(const State& state) const{
   std::vector<float> observation;
   state.ObservationTensor(player_id_,&observation);
   DominionObservation obs;

   //Decode tensor
   int offset = 0;
   for(int i = 0; i < kNumCards; i++){
    obs.cards_in_play.push_back(observation[offset]);
   }
   offset += kTreasureCards;
   for(int i = 0; i < kTreasureCards; i++){
    obs.treasure_supply.push_back(observation[offset]);
   }
   offset += kVictoryCards;
   for(int i = 0; i < kVictoryCards; i++){
    obs.victory_supply.push_back(observation[offset]);

   }
   offset += kKingdomCards;
   obs.phase = observation[offset];
   offset += 1;
   obs.actions = observation[offset];
   offset += 1;
   obs.buys = observation[offset];
   offset += 1;
   obs.coins = observation[offset];
   offset += 1;
   obs.effect = observation[offset];
   offset += 1;

   for(int i = 0; i < kHandSize; i++){
    obs.hand.push_back(observation[i]);
   }
   return obs;
  }
  
  Action BigMoneyBot::Step(const State& state) {
   DominionObservation obs = GetObservation(state);
   Action action;
   if(obs.phase == TurnPhase::TreasurePhase){
    action = play_first_treasure_card(obs.hand);
   }
   else if(obs.phase == TurnPhase::BuyPhase){
    if(obs.coins >= 3 && obs.coins <=5){
     action = purchase_treasure_card_if_avail(SILVER,obs.treasure_supply);
    }else if(obs.coins >= 6 && obs.coins <= 7){
     action = purchase_treasure_card_if_avail(GOLD,obs.treasure_supply);
    }
    else if(obs.coins >= 8){
     bool is_penultimate_province = obs.victory_supply[3] == 1;
     bool duchy_available = obs.victory_supply[2] > 0;
     if(duchy_dancing_ && is_penultimate_province && duchy_available){
      action = DUCHY.GetBuy();
     }else {
      action = PROVINCE.GetBuy();
     }
    return action;
   }
  }else{
   return state.LegalActions().front();
  }
 }
 }
}