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


void CellarEffect::Run(DominionState& state, PlayerState& PlayerState){
	state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
}

std::vector<Action> CellarEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const {
	std::set<Action> moves;
	for(const Card* card : PlayerState.GetHand()){
		moves.insert(card->GetDiscard());
	}
	moves.insert(END_PHASE_ACTION);
	std::vector<Action> legal_actions(moves.begin(),moves.end());
	absl::c_sort(legal_actions);
	return legal_actions;
}

void CellarEffect::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState) {
	if(action == END_PHASE_ACTION){
		state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
		PlayerState.DrawHand(num_cards_discarded_);
	}else{
		num_cards_discarded_ += 1;
		const Card* card = GetCard(action);
		PlayerState.RemoveFromPile(card,HAND);
		PlayerState.AddToPile(card,DISCARD);
	}
}

void TrashCardsEffect::Run(DominionState& state, PlayerState& PlayerState){
	state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
}

std::vector<Action> TrashCardsEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const {
	std::set<Action> moves;
	for(const Card* card : PlayerState.GetHand()){
		moves.insert(card->GetTrash());
	}
	moves.insert(END_PHASE_ACTION);
	std::vector<Action> legal_actions(moves.begin(),moves.end());
	absl::c_sort(legal_actions);
	return legal_actions;
}

void TrashCardsEffect::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState) {
	if(action == END_PHASE_ACTION){
		state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
	}else{
		const Card* card = GetCard(action);
		PlayerState.RemoveFromPile(card,HAND);
		PlayerState.AddToPile(card,TRASH);
		num_cards_trashed_ += 1;
		if (num_cards_trashed_ == num_cards_){
			state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
		}
	}
}

void OpponentsDiscardDownToEffect::Run(DominionState& state, PlayerState& player_state){
		for(PlayerState& player : state.getPlayers()){
			if(player.GetId() != player_state.GetId()){
				if(player.HasCardInHand(&MOAT)){
					state.GetEffectRunner()->AddEffect(this,player.GetId());
					}else{
						MILITIA_EFFECT_DISCARD.Run(state,player);
			}
		}
	}
}

std::vector<Action> OpponentsDiscardDownToEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const {
	return {MOAT.GetPlay(),END_PHASE_ACTION};
}

void OpponentsDiscardDownToEffect::DoApplyAction(Action action, DominionState& state, PlayerState& player) {
	if(action == END_PHASE_ACTION){
		MILITIA_EFFECT_DISCARD.Run(state,player);
	}else{
		state.GetEffectRunner()->RemoveEffect(player.GetId());
	}
}

void DiscardDownToEffect::Run(DominionState& state, PlayerState& player_state){
	state.GetEffectRunner()->AddEffect(this,player_state.GetId());
}

std::vector<Action> DiscardDownToEffect::LegalActions(const DominionState& state, const PlayerState& player) const {
	std::set<Action> moves;
	for(const Card* card : player.GetHand()){
		moves.insert(card->GetDiscard());
	}
	std::vector<Action> legal_actions(moves.begin(),moves.end());
	absl::c_sort(legal_actions);
	return legal_actions;
}

void DiscardDownToEffect::DoApplyAction(Action action, DominionState& state, PlayerState& player) {
	const Card* card = GetCard(action);
	player.RemoveFromPile(card,HAND);
	player.AddToPile(card,DISCARD);
	if(player.GetHand().size() == num_cards_down_to_){
		state.GetEffectRunner()->RemoveEffect(player.GetId());
	}
}

void OpponentsGainCardEffect::Run(DominionState& state, PlayerState& player_state){
		for(PlayerState& player : state.getPlayers()){
			if(player.GetId() != player_state.GetId()){
				if(player.HasCardInHand(&MOAT)){
					state.GetEffectRunner()->AddEffect(this,player.GetId());
				}else{
					WITCH_GAIN_EFFECT.Run(state,player);
				}
			}
	}
}

std::vector<Action> OpponentsGainCardEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const {
	return {MOAT.GetPlay(),END_PHASE_ACTION};
}

void OpponentsGainCardEffect::DoApplyAction(Action action, DominionState& state, PlayerState& player) {
	if(action == END_PHASE_ACTION){
		WITCH_GAIN_EFFECT.Run(state,player);
	}
	state.GetEffectRunner()->RemoveEffect(player.GetId());
}

void GainCardToDicardPileEffect::Run(DominionState& state, PlayerState& player_state){
	state.RemoveCardFromSupplyPile(card_->getName());
	player_state.AddToPile(card_,DISCARD);
}


void ChoosePileToGainEffect::Run(DominionState& state, PlayerState& PlayerState){
	state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
}

std::vector<Action> ChoosePileToGainEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const {
	std::vector<Action> legal_actions;
	for(auto pile : state.getSupplyPiles()){
		if(!pile.second.Empty() && pile.second.getCard()->GetCost() <= n_coins_){
			legal_actions.push_back(pile.second.getCard()->GetGain());
		}
	}
	absl::c_sort(legal_actions);
	return legal_actions;
}

void ChoosePileToGainEffect::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState) {
	const Card* card = GetCard(action);
	state.RemoveCardFromSupplyPile(card->getName());
	PlayerState.AddToPile(card,DISCARD);
	state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
}

}  // namespace dominion
}  // namespace open_spiel
