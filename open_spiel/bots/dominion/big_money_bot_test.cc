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

#include <ctime>
#include <memory>
#include <vector>

#include "open_spiel/bots/dominion/big_money_bot.h"
#include "open_spiel/games/dominion.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace dominion {
namespace bots {
namespace {

const GameParameters params =  {
  {"kingdom_cards",GameParameter(kDefaultKingdomCards)}
};

void BuysSilverWith3Coins() {
 int num_games = 1;
 std::mt19937 rng;
 auto game = LoadGame("dominion",params);
  std::vector<std::unique_ptr<Bot>> bots;
  for (Player p = 0; p < kNumPlayers; ++p) {
    bots.push_back(
        std::make_unique<BigMoneyBot>(game->GetParameters(), p,false));
  }
 auto state = game->NewInitialState();
 std::vector<Action> hand = {COPPER.GetId(), COPPER.GetId(), COPPER.GetId(),ESTATE.GetId(),ESTATE.GetId()};
 std::vector<Action> draw = {COPPER.GetId(), COPPER.GetId(), COPPER.GetId(),COPPER.GetId(),ESTATE.GetId()};
 for(int i = 0; i < kNumPlayers; i++){
  for(Action a : hand){
   state->ApplyAction(a);
  }
  for(Action a : draw){
   state->ApplyAction(a);
  }
 }
 std::vector<Action> actual_actions;
 std::vector<Action> empty;
 while(state->CurrentPlayer() == 0){
  Action action = bots[0]->Step(*state);
  state->ApplyAction(action);
  actual_actions.push_back(action);
 }
 std::vector<Action> expected = {COPPER.GetPlay(), COPPER.GetPlay(), COPPER.GetPlay(),END_PHASE_ACTION,SILVER.GetBuy(),END_PHASE_ACTION};
 SPIEL_CHECK_EQ(actual_actions,expected);
}
void BuysSilverWith5Coins() {
int num_games = 1;
 std::mt19937 rng;
 auto game = LoadGame("dominion",params);
  std::vector<std::unique_ptr<Bot>> bots;
  for (Player p = 0; p < kNumPlayers; ++p) {
    bots.push_back(
        std::make_unique<BigMoneyBot>(game->GetParameters(), p,false));
  }
 auto state = game->NewInitialState();
 std::vector<Action> hand = {COPPER.GetId(), COPPER.GetId(), COPPER.GetId(),COPPER.GetId(),COPPER.GetId()};
 std::vector<Action> draw = {COPPER.GetId(), COPPER.GetId(), ESTATE.GetId(),ESTATE.GetId(),ESTATE.GetId()};
 for(int i = 0; i < kNumPlayers; i++){
  for(Action a : hand){
   state->ApplyAction(a);
  }
  for(Action a : draw){
   state->ApplyAction(a);
  }
 }
 std::vector<Action> actual_actions;
 std::vector<Action> empty;
 while(state->CurrentPlayer() == 0){
  Action action = bots[0]->Step(*state);
  state->ApplyAction(action);
  actual_actions.push_back(action);
 }
 std::vector<Action> expected = {COPPER.GetPlay(), COPPER.GetPlay(), COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),END_PHASE_ACTION,SILVER.GetBuy(),END_PHASE_ACTION};
 SPIEL_CHECK_EQ(actual_actions,expected);
}
void BuysGoldWith6Coins() {
int num_games = 1;
 std::mt19937 rng;
 auto game = LoadGame("dominion",params);
  std::vector<std::unique_ptr<Bot>> bots;
  for (Player p = 0; p < kNumPlayers; ++p) {
    bots.push_back(
        std::make_unique<BigMoneyBot>(game->GetParameters(), p,false));
  }
 DominionState state(game,kDefaultKingdomCards);
 std::vector<Action> hand = {ESTATE.GetId(), ESTATE.GetId(), COPPER.GetId(),COPPER.GetId(),COPPER.GetId()};
 std::vector<Action> draw = {COPPER.GetId(), COPPER.GetId(), COPPER.GetId(),COPPER.GetId(),ESTATE.GetId()};
 for(int i = 0; i < kNumPlayers; i++){
  for(Action a : hand){
   state.DoApplyAction(a);
  }
  for(Action a : draw){
   state.DoApplyAction(a);
  }
 }
 std::vector<Action> actual_actions;
 std::vector<Action> empty;

 state.DoApplyAction(END_PHASE_ACTION);
 state.GetCurrentPlayerState().AddFrontToDrawPile(&SILVER);
 state.ApplyAction(END_PHASE_ACTION);
 SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
 state.DoApplyAction(END_PHASE_ACTION);
 state.ApplyAction(END_PHASE_ACTION);
 SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  while(state.CurrentPlayer() == 0){
    Action action = bots[0]->Step(state);
    state.DoApplyAction(action);
    actual_actions.push_back(action);
  }
 std::vector<Action> expected = {COPPER.GetPlay(), COPPER.GetPlay(), COPPER.GetPlay(),COPPER.GetPlay(),SILVER.GetPlay(),END_PHASE_ACTION,GOLD.GetBuy(),END_PHASE_ACTION};
 SPIEL_CHECK_EQ(actual_actions,expected);
}
void BuysGoldWith7Coins() {
int num_games = 1;
 std::mt19937 rng;
 auto game = LoadGame("dominion",params);
  std::vector<std::unique_ptr<Bot>> bots;
  for (Player p = 0; p < kNumPlayers; ++p) {
    bots.push_back(
        std::make_unique<BigMoneyBot>(game->GetParameters(), p,false));
  }
 DominionState state(game,kDefaultKingdomCards);
 std::vector<Action> hand = {ESTATE.GetId(), ESTATE.GetId(), COPPER.GetId(),COPPER.GetId(),COPPER.GetId()};
 std::vector<Action> draw = {COPPER.GetId(), COPPER.GetId(), COPPER.GetId(),COPPER.GetId(),ESTATE.GetId()};
 for(int i = 0; i < kNumPlayers; i++){
  for(Action a : hand){
   state.DoApplyAction(a);
  }
  for(Action a : draw){
   state.DoApplyAction(a);
  }
 }
 std::vector<Action> actual_actions;
 std::vector<Action> empty;

 state.DoApplyAction(END_PHASE_ACTION);
 state.GetCurrentPlayerState().AddFrontToDrawPile(&SILVER);
 state.GetCurrentPlayerState().AddFrontToDrawPile(&SILVER);
 state.ApplyAction(END_PHASE_ACTION);
 SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
 state.DoApplyAction(END_PHASE_ACTION);
 state.ApplyAction(END_PHASE_ACTION);
 SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  while(state.CurrentPlayer() == 0){
    Action action = bots[0]->Step(state);
    state.DoApplyAction(action);
    actual_actions.push_back(action);
  }
 std::vector<Action> expected = {COPPER.GetPlay(), COPPER.GetPlay(), COPPER.GetPlay(),SILVER.GetPlay(),SILVER.GetPlay(),END_PHASE_ACTION,GOLD.GetBuy(),END_PHASE_ACTION};
 SPIEL_CHECK_EQ(actual_actions,expected);
}

void BuysProvinceWith8Coins() {
int num_games = 1;
 std::mt19937 rng;
 auto game = LoadGame("dominion",params);
  std::vector<std::unique_ptr<Bot>> bots;
  for (Player p = 0; p < kNumPlayers; ++p) {
    bots.push_back(
        std::make_unique<BigMoneyBot>(game->GetParameters(), p,false));
  }
 DominionState state(game,kDefaultKingdomCards);
 std::vector<Action> hand = {ESTATE.GetId(), ESTATE.GetId(), COPPER.GetId(),COPPER.GetId(),COPPER.GetId()};
 std::vector<Action> draw = {ESTATE.GetId(), COPPER.GetId(), COPPER.GetId(),COPPER.GetId(),COPPER.GetId()};
 for(int i = 0; i < kNumPlayers; i++){
  for(Action a : hand){
   state.DoApplyAction(a);
  }
  for(Action a : draw){
   state.DoApplyAction(a);
  }
 }
 std::vector<Action> actual_actions;
 std::vector<Action> empty;

 state.DoApplyAction(END_PHASE_ACTION);
 for(int i = 0; i < 4; i++){
   state.GetCurrentPlayerState().AddFrontToDrawPile(&SILVER);
 }
 state.ApplyAction(END_PHASE_ACTION);
 SPIEL_CHECK_EQ(state.CurrentPlayer(),1);
 state.DoApplyAction(END_PHASE_ACTION);
 state.ApplyAction(END_PHASE_ACTION);
 SPIEL_CHECK_EQ(state.CurrentPlayer(),0);
  while(state.CurrentPlayer() == 0){
    Action action = bots[0]->Step(state);
    state.DoApplyAction(action);
    actual_actions.push_back(action);
  }
 std::vector<Action> expected = {SILVER.GetPlay(),SILVER.GetPlay(),SILVER.GetPlay(),SILVER.GetPlay(),END_PHASE_ACTION,PROVINCE.GetBuy(),END_PHASE_ACTION};
 SPIEL_CHECK_EQ(actual_actions,expected);
}
void BigMoneyBotBeatsRandom(){
 std::mt19937 rng;
 auto game = LoadGame("dominion",params);
 auto player_0_bot = std::make_unique<BigMoneyBot>(game->GetParameters(), 0,false);
std::unique_ptr<State> state = game->NewInitialState();
while(!state->IsTerminal()){
  if(state->IsChanceNode()){
    Action outcome = SampleAction(state->ChanceOutcomes(),
                      std::uniform_real_distribution<double>(0.0, 1.0)(rng))
            .first;
  state->ApplyAction(outcome);
  }else{
    auto curr_player = state->CurrentPlayer();
    Action action;
    if(curr_player == 0){
      action = player_0_bot->Step(*state);
    }else{
      std::vector<Action> legal_actions = state->LegalActions();
      double prob = 1.0 / size(legal_actions);
      ActionsAndProbs ActionsAndProbs = {};
      for(Action a: legal_actions){
        ActionsAndProbs.push_back(std::make_pair(prob,a));
      }
      action = SampleAction(ActionsAndProbs,std::uniform_real_distribution<double>(0.0,1.0)(rng)).first;
    }
    state->ApplyAction(action);
  }
}
  std::vector<double> expected_returns = {1.0,0.0};
  SPIEL_CHECK_EQ(state->Returns(),expected_returns);
}



} // namespace
}  // namespace bots
}  // namespace dominion
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::dominion::bots::BuysSilverWith3Coins();
  open_spiel::dominion::bots::BuysSilverWith5Coins();
  open_spiel::dominion::bots::BuysGoldWith6Coins();
  open_spiel::dominion::bots::BuysGoldWith7Coins();
  open_spiel::dominion::bots::BuysProvinceWith8Coins();
  open_spiel::dominion::bots::BigMoneyBotBeatsRandom();
}
