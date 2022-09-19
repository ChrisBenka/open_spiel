// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <vector>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/spiel_bots.h"

#include "open_spiel/bots/dominion/big_money_bot.h"
#include "open_spiel/games/dominion.h"

namespace open_spiel {
namespace dominion {
namespace bots {

Action BigMoneyBot::PlayFirstTreasureCard(const std::vector<int> hand) const {
   auto ptr = absl::c_find_if(hand,[](const int id){
    return GetCard(id)->getCardType() == TREASURE;
   });
   if(ptr == hand.end()){
    return END_PHASE_ACTION;
   }else{
    int id = *ptr;
    const Card* c = GetCard(id);
    return c->GetPlay();
   }
}

Action BigMoneyBot::PurchaseTreasureCardIfAvail(const TreasureCard card, const std::vector<int> treasure_supply) const {
  int treasure_offset = COPPER.GetId();
  return treasure_supply[card.GetId()-treasure_offset] > 0 ? card.GetBuy() : END_PHASE_ACTION;
}

Action BigMoneyBot::Step(const State& state) {
   DominionObservation obs = GetObservation(state);
   Action action;
   if(obs.phase == TurnPhase::TreasurePhase){
    return PlayFirstTreasureCard(obs.hand);
   }
   else if(obs.phase == TurnPhase::BuyPhase){
    if(obs.coins < 3){
      return END_PHASE_ACTION;
    }
   else if(obs.coins >= 3 && obs.coins <=5){
     return PurchaseTreasureCardIfAvail(SILVER,obs.treasure_supply);
    }else if(obs.coins >= 6 && obs.coins <= 7){
     return PurchaseTreasureCardIfAvail(GOLD,obs.treasure_supply);
    }
    else if(obs.coins >= 8){
     bool is_penultimate_province = obs.victory_supply[3] == 1;
     bool duchy_available = obs.victory_supply[2] > 0;
     if(duchy_dancing_ && is_penultimate_province && duchy_available){
      return DUCHY.GetBuy();
     }else {
      return PROVINCE.GetBuy();
     }
   }
  }else{
   return state.LegalActions().front();
  }
}

std::pair<ActionsAndProbs, Action> BigMoneyBot::StepWithPolicy(const State& state) {
  ActionsAndProbs policy;
  auto legal_actions = state.LegalActions(player_id_);
  auto chosen_action = Step(state);
  for (auto action : legal_actions)
    policy.emplace_back(action, action == chosen_action ? 1.0 : 0.0);
  return std::make_pair(policy,chosen_action);
}

ActionsAndProbs BigMoneyBot::GetPolicy(const State& state) {
  ActionsAndProbs policy;
  auto legal_actions = state.LegalActions(player_id_);
  auto chosen_action = Step(state);
  for (auto action : legal_actions)
    policy.emplace_back(action, action == chosen_action ? 1.0 : 0.0);
  return policy;
}

DominionObservation BigMoneyBot::GetObservation(const State& state) const{
   std::vector<float> observation;
   state.ObservationTensor(player_id_,&observation);
   DominionObservation obs;
   //Decode tensor
   int offset = 0;
   for(int i = 0; i < kNumCards; i++){
    obs.cards_in_play.push_back(observation[offset + i]);
   }
   offset += kNumCards;
   for(int i = 0; i < kTreasureCards; i++){
    obs.treasure_supply.push_back(observation[offset + i]);
   }
   offset += kTreasureCards;
   for(int i = 0; i < kVictoryCards; i++){
    obs.victory_supply.push_back(observation[offset + i]);
   }
  offset += kVictoryCards;

  for(int i = 0; i < kKingdomCards; i++){
    obs.kingdom_supply.push_back(observation[offset + i]);
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
  obs.hand.push_back(observation[offset + i]);
  }
  return obs;
  }

}   //namespace bots
}  // namespace dominion
}  // namespace open_spiel

