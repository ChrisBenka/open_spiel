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
namespace {

const GameParameters params =  {
  {"kingdom_cards",GameParameter(kDefaultKingdomCards)}
};

void BuysSilverWith3Coins() {
 int num_games = 1;
 std::mt19937 rng;
 auto game = LoadGame("dominion",params);
 std::vector<std::unique_ptr<Bot>> bots = {
  std::make_unique<BigMoneyBot>(game->GetParameters(),0,false),
  std::make_unique<BigMoneyBot>(game->GetParameters(),1,false),
 };
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
 std::vector<std::unique_ptr<Bot>> bots = {
  std::make_unique<BigMoneyBot>(game->GetParameters(),0,false),
  std::make_unique<BigMoneyBot>(game->GetParameters(),1,false),
 };
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
 while(state->CurrentPlayer() == 0){
  Action action = bots[0]->Step(*state);
  state->ApplyAction(action);
  actual_actions.push_back(action);
 }
 std::vector<Action> expected = {COPPER.GetPlay(), COPPER.GetPlay(), COPPER.GetPlay(),COPPER.GetPlay(),COPPER.GetPlay(),END_PHASE_ACTION,SILVER.GetBuy()};
 SPIEL_CHECK_EQ(actual_actions,expected);
}


}  // namespace
}  // namespace dominion
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::dominion::BuysSilverWith3Coins();
}
