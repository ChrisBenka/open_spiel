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

void GainTreasureCardEffect::Run(DominionState& state, PlayerState& PlayerState){
	if(!state.getSupplyPiles().at(card_->getName()).Empty()){
		state.RemoveCardFromSupplyPile(card_->getName());
		PlayerState.AddToPile(card_,DISCARD);
	}
}


int OpponentRevalAndTrashTopTwoCards::NumberTrashableCards() const {
	return absl::c_count_if(top_two_,[](const Card* c){
		return c->getCardType() == TREASURE && c->GetId() != COPPER.GetId();
	});
}
bool OpponentRevalAndTrashTopTwoCards::isTrashable(const Card* c) const {
	return c->getCardType() == TREASURE && c->GetId() != COPPER.GetId();
}

std::vector<Action> OpponentRevalAndTrashTopTwoCards::LegalActions(const DominionState& state, const PlayerState& PlayerState) const {
	std::set<Action> moves = {};
	if(NumberTrashableCards() == 2){
		moves = {top_two_.at(0)->GetTrash(),top_two_.at(1)->GetTrash()};
	}else if(NumberTrashableCards() == 1 && top_two_.size() == 2){
		if(isTrashable(top_two_.at(0))){
		moves = {top_two_.at(0)->GetTrash()};
		}else{
			moves = {top_two_.at(1)->GetTrash()};
		}
	}else if(top_two_.size() == 1){
		moves = {top_two_.front()->GetDiscard()};
	}
	else{
		moves.insert(top_two_.at(0)->GetDiscard());
		moves.insert(top_two_.at(1)->GetDiscard());
	}
	std::vector<Action> legal_actions(moves.begin(),moves.end());
	absl::c_sort(legal_actions);
	return legal_actions;
}

void OpponentRevalAndTrashTopTwoCards::Run(DominionState& state, PlayerState& player_state){
	if(player_state.GetDrawPile().size() < 2){
			player_state.SetAddDiscardPileToDrawPile(true);
			player_state.SetNumRequiredCards(2 - player_state.GetDrawPile().size());
	}
	state.GetEffectRunner()->AddEffect(this,player_state.GetId());
}

void BanditEffect::Run(DominionState& state, PlayerState& player_state){
		GAIN_GOLD.Run(state,player_state);
		for(PlayerState& p : state.getPlayers()){
			if(p.GetId() != player_state.GetId()){
					TRASH_TOP_TWO.Run(state,p);
			}
		}
}

void OpponentRevalAndTrashTopTwoCards::DoApplyAction(Action action, DominionState& state, PlayerState& player_state){
		const Card* card = GetCard(action);
		auto it = absl::c_find_if(top_two_,[card](const Card* c){
    return card->GetId() == c->GetId(); 
  });
  top_two_.erase(it);
		if(action == card->GetTrash()){
			player_state.AddToPile(card,TRASH);
		}else{
			player_state.AddToPile(card,DISCARD);
			state.GetEffectRunner()->RemoveEffect(player_state.GetId());
		}
}

void OpponentRevalAndTrashTopTwoCards::DoPostApplyChanceOutcomeAction(DominionState& state, PlayerState& player_state){
		const Card* card1 = player_state.TopOfDeck();
		player_state.RemoveFromPile(card1,DRAW);
		const Card* card2 = player_state.TopOfDeck();
		player_state.RemoveFromPile(card2,DRAW);
		top_two_ = {card1, card2};
}

void TrashAndGainEffect::Run(DominionState& state, PlayerState& PlayerState){
	state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
}
std::vector<Action> TrashAndGainEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const {
	std::set<Action> moves;
	if(trashed_card_ == nullptr){
		for(const Card* card : PlayerState.GetHand()){
			moves.insert(card->GetTrash());
		}
	}else{
		for(auto pile_pair : state.getSupplyPiles()){
			if(!pile_pair.second.Empty() && 
			pile_pair.second.getCard()->GetCost() <= trashed_card_->GetCost() + n_coins_){
				moves.insert(pile_pair.second.getCard()->GetGain());
			}
		}
	}
	std::vector<Action> legal_actions(moves.begin(),moves.end());
	absl::c_sort(legal_actions);
	return legal_actions;
}
void TrashAndGainEffect::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState) {
	const Card* card = GetCard(action);
	if(trashed_card_ == nullptr){
		trashed_card_ = card;
		PlayerState.RemoveFromPile(card,HAND);
		PlayerState.AddToPile(card,TRASH);
	}else{
		PlayerState.AddToPile(card,DISCARD);
		state.RemoveCardFromSupplyPile(card->getName());
	}
}

void TrashCardAndGainCoins::Run(DominionState& state, PlayerState& PlayerState){
	state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
}
std::vector<Action> TrashCardAndGainCoins::LegalActions(const DominionState& state, const PlayerState& PlayerState) const {
	if(PlayerState.HasCardInHand(card_to_trash_)){
		return {card_to_trash_->GetTrash(),END_PHASE_ACTION};
	}else{
		return {END_PHASE_ACTION};
	}
}
void TrashCardAndGainCoins::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState) {
	if(action == card_to_trash_->GetTrash()){
		std::cout << n_coins_;
		PlayerState.addCoins(n_coins_);
		PlayerState.AddToPile(card_to_trash_,TRASH);
		PlayerState.RemoveFromPile(card_to_trash_,HAND);
	}
	state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
}

void PoacherEffect::Run(DominionState& state, PlayerState& PlayerState){
	num_empty_piles_ = absl::c_count_if(state.getSupplyPiles(),[](std::pair<std::string,SupplyPile> pile_pair){
		return pile_pair.second.Empty();
	});
	if(num_empty_piles_ != 0){
		state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
	}
}

void PoacherEffect::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState){
	const Card* card = GetCard(action);
	PlayerState.RemoveFromPile(card,HAND);
	PlayerState.AddToPile(card,DISCARD);
	num_cards_discarded_ += 1;
	if (num_cards_discarded_ == num_empty_piles_){
		state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
	}
}

 std::vector<Action> PoacherEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const{
	std::set<Action> moves;
	for(const Card* card : PlayerState.GetHand()){
		moves.insert(card->GetDiscard());
	}
	std::vector<Action> legal_actions(moves.begin(),moves.end());
	absl::c_sort(legal_actions);
	return legal_actions;
}

void TrashTreasureAndGainTreasure::Run(DominionState& state, PlayerState& PlayerState){
	if(PlayerState.HasTreasureCardInHand()){
		state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
	}
}

void TrashTreasureAndGainTreasure::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState){
	const Card* card = GetCard(action);
	if(action == card->GetTrash()){
		trashed_card_ = card;
		PlayerState.RemoveFromPile(card,HAND);
		PlayerState.AddToPile(card,TRASH);
	}else if(action == card->GetGain()){
		PlayerState.AddToPile(card,DISCARD);
		state.RemoveCardFromSupplyPile(card->getName());
		state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
	}else{
		state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
	}
}

 std::vector<Action> TrashTreasureAndGainTreasure::LegalActions(const DominionState& state, const PlayerState& PlayerState) const{
	std::set<Action> moves;
	if(trashed_card_ == nullptr){
		for(const Card* card : PlayerState.GetHand()){
			if(card->getCardType() == TREASURE){
					moves.insert(card->GetTrash());
			}
		}
		moves.insert(END_PHASE_ACTION);
	}else {
		for(auto pile_pair_ : state.getSupplyPiles()){
			if(!pile_pair_.second.Empty() && 
			pile_pair_.second.getCard()->GetCost() <= trashed_card_->GetCost() + n_coins_
			&& pile_pair_.second.getCard()->getCardType() == TREASURE){
				moves.insert(pile_pair_.second.getCard()->GetGain());
			}
		}
	}
	std::vector<Action> legal_actions(moves.begin(),moves.end());
	absl::c_sort(legal_actions);
	return legal_actions;
}



void VassalEffect::Run(DominionState& state, PlayerState& PlayerState){
	state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
}

void VassalEffect::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState){
	state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
	const Card* card = GetCard(action);
	if(action == card->GetDiscard()){
		PlayerState.AddToPile(card,DISCARD);
		PlayerState.RemoveFromPile(card,DRAW);
	}else{
		PlayerState.PlayActionCard(state,card,DRAW);
	}
}

 std::vector<Action> VassalEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const{
		if(PlayerState.GetDrawPile().front()->getCardType() == ACTION){
			return {PlayerState.GetDrawPile().front()->GetPlay(),PlayerState.GetDrawPile().front()->GetDiscard()};
		}else{
			return {PlayerState.GetDrawPile().front()->GetDiscard()};
		}
}

void DrawCardsEffect::Run(DominionState& state, PlayerState& player){
	for(PlayerState& p : state.getPlayers()){
		if(p.GetId() != player.GetId()){
			p.DrawHand(n_cards_);
		}
	}
}


void ArtisanEffect::Run(DominionState& state, PlayerState& PlayerState){
	state.GetEffectRunner()->AddEffect(this,PlayerState.GetId());
}

void ArtisanEffect::DoApplyAction(Action action, DominionState& state, PlayerState& PlayerState){
	const Card* card = GetCard(action);
	if(card_gained_ == nullptr){
		state.RemoveCardFromSupplyPile(card->getName());
		PlayerState.AddToPile(card,HAND);
		card_gained_ = card;
	}else{
		state.GetEffectRunner()->RemoveEffect(PlayerState.GetId());
		PlayerState.RemoveFromPile(card,HAND);
		PlayerState.AddFrontToDrawPile(card);
		
	}
}

std::vector<Action> ArtisanEffect::LegalActions(const DominionState& state, const PlayerState& PlayerState) const{
	if(card_gained_ == nullptr){
		std::vector<Action> legal_actions;
		for(auto pile : state.getSupplyPiles()){
			if(!pile.second.Empty() && pile.second.getCard()->GetCost() <= n_coins_){
				legal_actions.push_back(pile.second.getCard()->GetGain());
			}
		}
		absl::c_sort(legal_actions);
		return legal_actions;
	}else{
		std::set<Action> moves;
		for(const Card* card: PlayerState.GetHand()){
			moves.insert(card->GetPlaceOntoDeck());
		}
		std::vector<Action> legal_actions(moves.begin(),moves.end());
		absl::c_sort(legal_actions);
		return legal_actions;
	}
}

}  // namespace dominion
}  // namespace open_spiel
