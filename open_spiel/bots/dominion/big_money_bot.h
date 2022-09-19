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

#ifndef OPEN_SPIEL_BOTS_DOMINION_H_
#define OPEN_SPIEL_BOTS_DOMINION_H_

#include <vector>

#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/games/dominion.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_bots.h"

namespace open_spiel {
namespace dominion {
namespace bots {
class BigMoneyBot : public Bot {
  public:
    BigMoneyBot(GameParameters params, const Player player_id, bool duchy_dancing) : 
     params_(params), player_id_(player_id), duchy_dancing_(duchy_dancing) {}
    void Restart() {};
    Action Step(const State& state) override;
    bool ProvidesPolicy() override { return true; }
    ActionsAndProbs GetPolicy(const State& state) override;
    std::pair<ActionsAndProbs, Action> StepWithPolicy(const State& state) override;
  private:
    Action PlayFirstTreasureCard(const std::vector<int> hand) const;
    Action PurchaseTreasureCardIfAvail(const TreasureCard card,const std::vector<int> treasure_supply) const;
    DominionObservation GetObservation(const State& state) const;
    GameParameters params_;
    const Player player_id_;
    bool duchy_dancing_;  
};
}  // namespace bots
}  // namespace dominion
}  // namespace open_spiel

#endif  // OPEN_SPIEL_BOTS_DOMINION_H_

