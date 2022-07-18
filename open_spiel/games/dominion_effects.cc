// // Copyright 2022 DeepMind Technologies Limited
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //      http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.


#include "open_spiel/games/dominion.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <map>
#include <string>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"

namespace open_spiel {
namespace dominion {

using namespace dominion;

bool EffectRunner::Active() const {
 return absl::c_find(effects_,[](const Effect* effect){
  effect != nullptr;
 });
}
Player EffectRunner::CurrentPlayer() const {
 auto effect_itr =  absl::c_find(effects_,[](const Effect* effect){
  effect != nullptr;
 });
 effect_itr - effects_.begin();
}

void CellarEffect::Run(std::shared_ptr<DominionState> state, PlayerState& PlayerState){
	state->GetEffectRunner().AddEffect(this,PlayerState.GetId());
}

std::vector<Action> CellarEffect::LegalActions(std::shared_ptr<DominionState> state, PlayerState& PlayerState) const {
	std::vector<Action> moves;
	for(const Card* card : PlayerState.GetHand()){
		moves.push_back(card->GetDiscard());
	}
	absl::c_sort(moves);
	return moves;
};

void CellarEffect::DoApplyAction(Action action,std::shared_ptr<DominionState> state, PlayerState& PlayerState) {
	if(action == END_PHASE_ACTION){
		state->GetEffectRunner().RemoveEffect(PlayerState.GetId());
		PlayerState.DrawHand(num_cards_discarded_);
	}else{
		const Card* card = GetCard(action);
		PlayerState.RemoveFromPile(card,HAND);
		PlayerState.AddToDrawPile(card);
		num_cards_discarded_ += 1;
	}
};




}  // namespace dominion
}  // namespace open_spiel
