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
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"dominion",
    /*long_name=*/"Dominion",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{
      {"kingdom_cards",GameParameter(kDefaultKingdomCards)}
    }
   };

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new DominionGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

const char* GetTurnPhaseStrings(int enumVal){
  return TurnPhaseStrings[enumVal];
}

Card::Card(int id, std::string name, int cost)
{
  id_ = id;
  name_ = name;
  play_ =  id;
  buy_ = 2 * kNumCards + id;
  trash_ = 3 * kNumCards + id;
  discard_ = 4 * kNumCards + id;
  gain_ = 5 * kNumCards + id;
  reveal_ = 6 * kNumCards + id;
  cost_ = cost;
  action_strs_ = {
    {play_, "Play"},
    {buy_, "Buy"},
    {trash_, "Trash"},
    {discard_, "Discard"},
    {gain_, "Gain"},
    {reveal_, "Reveal"}
  };
}

void PlayerState::DrawHand(int num_cards){
  add_discard_pile_to_draw_pile_ = draw_pile_.size() < num_cards;
  if(add_discard_pile_to_draw_pile_){
    num_required_cards_ = num_cards;
  }else {
    std::list<const Card*>::iterator it(draw_pile_.begin());
    std::advance(it,num_cards);
    hand_.splice(hand_.end(),draw_pile_,draw_pile_.begin(),it);
  }
}

void PlayerState::PlayTreasureCard(const Card* card){
  addCoins(card->GetCoins());
  cards_in_play_.push_back(card);
  auto it = absl::c_find(hand_,card);
  hand_.erase(it);
}

void PlayerState::PlayActionCard(DominionState& state, const Card* card){
  cards_in_play_.push_back(card);
  auto it = absl::c_find_if(hand_,[card](const Card* c){
    return card->GetId() == c->GetId(); 
  });
  hand_.erase(it);
  actions_ -= 1;
  actions_ += card->GetAddActions();
  buys_ += card->GetAddBuys();
  coins_ += card->GetCoins();
  if(card->GetAddCards() != 0){
    DrawHand(card->GetAddCards());
  }
  if(!(card->GetEffect() == nullptr)){
    card->GetEffect()->Run(state,*this);
  }
}

int PlayerState::GetVictoryPoints() const {
  std::list<const Card*> cards = GetAllCards();
  return absl::c_accumulate(cards,0,[cards](int vp, const Card* card){
    return card->getCardType() == VICTORY ? 
      card->GetVictoryPointsFn() ? vp + card->GetVictoryPointsFn()(cards) : 
      vp + card->GetVictoryPoints()
      : vp;
  });
}

void PlayerState::BuyCard(const Card* card){
  discard_pile_.push_back(card);
  coins_ -= card->GetCost();
  buys_ -= 1;
}

void PlayerState::AddToDrawPile(const Card* card){
  draw_pile_.push_back(card);

}
void PlayerState::RemoveFromPile(const Card* card, PileType pile){
  std::list<const Card*>& cards_ = pile == DISCARD ? 
    discard_pile_ :  pile == DRAW ? draw_pile_ :
    pile == HAND ? hand_ : trash_pile_;
  auto it = absl::c_find(cards_,card);
  if(it != cards_.end()){
    cards_.erase(it);
  }
}

std::list<const Card*> PlayerState::GetAllCards() const {
  std::list<const Card*> all_player_cards;
  all_player_cards.insert(all_player_cards.end(),draw_pile_.begin(),draw_pile_.end());
  all_player_cards.insert(all_player_cards.end(),hand_.begin(),hand_.end());
  all_player_cards.insert(all_player_cards.end(),discard_pile_.begin(),discard_pile_.end());
  all_player_cards.insert(all_player_cards.end(),trash_pile_.begin(),trash_pile_.end());
  all_player_cards.insert(all_player_cards.end(),cards_in_play_.begin(),cards_in_play_.end());
  return all_player_cards;
}

TurnPhase PlayerState::EndPhase(){
  switch (turn_phase_)
  {
  case ActionPhase:
    turn_phase_ = TreasurePhase;
    break;
    
  case TreasurePhase:
    turn_phase_ = BuyPhase;
    break;
  default:
    turn_phase_ = TurnPhase::EndTurn;
  }
  return turn_phase_;
}

void PlayerState::AddHandInPlayCardsToDiscardPile() {
  discard_pile_.splice(discard_pile_.end(),hand_);
  discard_pile_.splice(discard_pile_.end(),cards_in_play_);
}

bool PlayerState::HasActionCardsInHand() const {
  return absl::c_find_if(hand_,[](const Card* card){
    return card->getCardType() == CardType::ACTION;
  }) != hand_.end();
}

void PlayerState::EndTurn() {
	AddHandInPlayCardsToDiscardPile();
	DrawHand(kHandSize);
	actions_ = 1;
	buys_ = 1;
	coins_ = 0;
	turn_phase_ = HasActionCardsInHand() ? ActionPhase : TreasurePhase;
}



std::map<std::string,SupplyPile> createTreasurePiles(){
  std::map<std::string,SupplyPile>treasurePiles;
  SupplyPile copper_supply(&COPPER,46);
  SupplyPile silver_supply(&SILVER,40);
  SupplyPile gold_supply(&GOLD,30);
  treasurePiles.insert(std::pair<std::string,SupplyPile>("Copper",copper_supply));;
  treasurePiles.insert(std::pair<std::string,SupplyPile>("Silver",silver_supply));;
  treasurePiles.insert(std::pair<std::string,SupplyPile>("Gold",gold_supply));;
  return treasurePiles;
}

std::map<std::string,SupplyPile> createVictoryPiles(){
  std::map<std::string,SupplyPile>victoryPiles;
  SupplyPile curse_supply(&CURSE,10);
  SupplyPile estate_supply(&ESTATE,8);
  SupplyPile duchy_supply(&DUCHY,8);
  SupplyPile province_supply(&PROVINCE,8);

  victoryPiles.insert(std::pair<std::string,SupplyPile>("Curse",curse_supply));
  victoryPiles.insert(std::pair<std::string,SupplyPile>("Estate",estate_supply));
  victoryPiles.insert(std::pair<std::string,SupplyPile>("Duchy",duchy_supply));
  victoryPiles.insert(std::pair<std::string,SupplyPile>("Province",province_supply));

  return victoryPiles;
}


std::map<std::string,SupplyPile> createKingdomPiles(std::vector<std::string> kingdom_cards){
  std::vector<const Card*> cards;
  for(const std::string card_nm : kingdom_cards){
    auto card_itr = absl::c_find_if(all_cards,[card_nm](const Card* card){return card->getName().compare(card_nm) == 0;});
    cards.push_back(*(card_itr));
  }
  std::map<std::string,SupplyPile> kingdom_piles;
  for(const Card* card : cards){
    int qty = card->getName() == GARDENS.getName() ? kGardenSupply : kInitSupply;
    std::pair<std::string,SupplyPile> pile = {card->getName(),SupplyPile(card,qty)};
    kingdom_piles.insert(pile);
  }
  return kingdom_piles;
}

void DominionState::DoApplyPlayTreasureCard(const Card* card){
  players_[current_player_].PlayTreasureCard(card);
}

void DominionState::DoApplyBuyCard(const Card* card){
  players_[current_player_].BuyCard(card);
  supply_piles_.find(card->getName())->second.RemoveCardFromSupplyPile();
}

void DominionState::DoApplyEndPhaseAction(){
  TurnPhase updated_phase = players_[current_player_].EndPhase();
  if(updated_phase == EndTurn){
    players_[current_player_].EndTurn();
    MoveToNextPlayer();
  }
}

void DominionState::MoveToNextPlayer(){
  current_player_ = (current_player_ + 1) % players_.size();
}

void DominionState::DoApplyAction(Action action_id) {
  if(IsChanceNode()){
    DoApplyChanceAction(action_id);
  }else{
    std::vector<Action> legal_actions = LegalActions();
    SPIEL_CHECK_EQ(absl::c_count(legal_actions,action_id),1);

    if(effect_runner_->Active()){
      Player player = effect_runner_->CurrentPlayer();
      effect_runner_->GetEffect(player)->DoApplyAction(action_id,*this,players_.at(player));
    }

    if(action_id == END_PHASE_ACTION){
      DoApplyEndPhaseAction();
    }else{
      const Card* card = GetCard(action_id);
      switch (players_[current_player_].GetTurnPhase())
      {
      case TreasurePhase:
        players_[current_player_].PlayTreasureCard(card);
        break;
      case BuyPhase:
        DoApplyBuyCard(card);
        break;
      case ActionPhase:
        players_[current_player_].PlayActionCard(*this,card);
        break;
      default:
        break;
      }
    }
  }
  is_terminal_ = GameFinished();
}

std::vector<std::pair<Action, double>> DominionState::GetInitSupplyChanceOutcomes() const {
  const std::list<const Card*> draw_pile = players_.at(current_player_).GetDrawPile(); 
  const int num_coppers = absl::c_count_if(draw_pile,[](const Card* card){return card->GetId() == COPPER.GetId();});
  const int num_estates = absl::c_count_if(draw_pile,[](const Card* card){return card->GetId() == ESTATE.GetId();});
  const double total = kInitSupply - (num_coppers + num_estates);
  const double copper_p = (kInitCoppers - num_coppers) / total;
  const double estate_p = (kInitEstates - num_estates) / total;
  return std::vector<std::pair<Action,double>>{
    std::pair<Action,double>{COPPER.GetId(),copper_p},
    std::pair<Action,double>{ESTATE.GetId(),estate_p},
  };
}

std::vector<std::pair<Action, double>> DominionState::GetAddDiscardPileToDrawPileChanceOutcomes() const { 
  auto player = absl::c_find_if(players_,[](PlayerState playerState) {
    return playerState.GetAddDiscardPileToDrawPile();
  });
  std::list<const Card*> discard_pile = player->GetDiscardPile();
  std::set<const Card*> unique_discard_pile(discard_pile.begin(),discard_pile.end());
  std::unordered_map<int,int> counter;
  for(auto card : discard_pile){
    if(counter.find(card->GetId()) == counter.end()) counter[card->GetId()] = 1;
    else counter[card->GetId()]++;
  }
  std::vector<std::pair<Action,double>> outcomes;
  double total_cards = discard_pile.size();
  for(auto card : unique_discard_pile){
    int card_count = counter.find(card->GetId())->second;
    outcomes.push_back(std::pair<Action,double>(card->GetId(),card_count/total_cards));
  }
  absl::c_sort(outcomes,[](std::pair<Action,double> outcome1, std::pair<Action,double> outcome2){
    return outcome1.first < outcome2.first;
  });
  return outcomes;
}

std::vector<std::pair<Action, double>> DominionState::ChanceOutcomes() const {
  SPIEL_CHECK_EQ(CurrentPlayer(),kChancePlayerId);
  if(!EachPlayerReceivedInitSupply()){
    return GetInitSupplyChanceOutcomes();
  }else if(AddDiscardPileToDrawPile()){
    return GetAddDiscardPileToDrawPileChanceOutcomes();
  }
 }

 std::vector<Action> DominionState::LegalTreasurePhaseActions() const {
  std::list<const Card*> hand = players_.at(current_player_).GetHand();
  std::unordered_set<Action> moves;
  absl::c_for_each(hand,[&moves](const Card* card){
    if(card->getCardType() == TREASURE)
      moves.insert(card->GetPlay());
  });
  moves.insert(END_PHASE_ACTION);
  std::vector<Action> legal_actions(moves.begin(),moves.end());
  absl::c_sort(legal_actions);
  return legal_actions;
 }

 std::vector<Action> DominionState::LegalBuyPhaseActions() const {
  int coins = players_.at(current_player_).GetCoins();
  std::unordered_set<Action> moves;
  absl::c_for_each(supply_piles_,[&moves,coins](std::pair<std::string,SupplyPile> pile){
    if(!pile.second.Empty() && pile.second.getCard()->GetCost() <= coins){
      moves.insert(pile.second.getCard()->GetBuy());
    }
  });
  moves.insert(END_PHASE_ACTION);
  std::vector<Action> legal_actions(moves.begin(),moves.end());
  absl::c_sort(legal_actions);
  return legal_actions;
 }

 std::vector<Action> DominionState::LegalActionPhaseActions() const {
  if(players_.at(current_player_).GetActions() == 0) return {END_PHASE_ACTION};
  std::list<const Card*> hand = players_.at(current_player_).GetHand();
  std::unordered_set<Action> moves;
  absl::c_for_each(hand,[&moves](const Card* card){
     if(card->getCardType() == ACTION)
      moves.insert(card->GetPlay());
  });
  moves.insert(END_PHASE_ACTION);
  std::vector<Action> legal_actions(moves.begin(),moves.end());
  absl::c_sort(legal_actions);
  return legal_actions;
 }

Player DominionState::CurrentEffectPlayer() const{
  for(int i = 0; i < kNumPlayers; i++){
    if(effects_[i] != nullptr){
      return i;
    }
  }
}

std::vector<Action> DominionState::LegalActions() const {
  // if (IsTerminal()) return {};
  SPIEL_CHECK_GE(current_player_,0);
  if(effect_runner_->Active()){
    Player player = effect_runner_->CurrentPlayer();
    return effect_runner_->GetEffect(player)->LegalActions(*this,players_.at(player));
  }
  switch (players_.at(current_player_).GetTurnPhase()){
  case TreasurePhase:
    return LegalTreasurePhaseActions();
  case BuyPhase:
    return LegalBuyPhaseActions();
  case ActionPhase:
    return LegalActionPhaseActions();
  default:
    return {};
  }
}

std::string DominionState::ActionToString(Player player,
                                           Action action_id) const {
  std::string str;
  if(action_id == END_PHASE_ACTION){
    TurnPhase phase = players_.at(current_player_).GetTurnPhase();
    absl::StrAppend(&str,"End ",std::string(GetTurnPhaseStrings(phase)));
  }else{
    const Card* card = GetCard(action_id);
    absl::StrAppend(&str,card->ActionToString(action_id)," ",card->getName());
  }
  return str;
}

std::vector<std::string> splitString(std::string str, char splitter){
    std::vector<std::string> result;
    std::string current = ""; 

    for(int i = 0; i < str.size(); i++){
        if(str[i] == splitter){
            if(current != ""){
                result.push_back(current);
                current = "";
            } 
            continue;
        }
        current += str[i];
    }
    if(current.size() != 0)
        result.push_back(current);
    return result;
}

DominionState::DominionState(std::shared_ptr<const Game> game, std::string kingdom_cards) : 
kingdom_cards_(splitString(kingdom_cards,';')), effect_runner_(new EffectRunner()) ,State(game) {
  std::map<std::string,SupplyPile> victory_piles = createVictoryPiles();
  std::map<std::string,SupplyPile> treasure_piles = createTreasurePiles();
  if(kingdom_cards.length() != 0){
    std::map<std::string,SupplyPile> kingdom_piles = createKingdomPiles(kingdom_cards_);
    supply_piles_.insert(kingdom_piles.begin(),kingdom_piles.end());
  }
  supply_piles_.insert(victory_piles.begin(),victory_piles.end());
  supply_piles_.insert(treasure_piles.begin(),treasure_piles.end());
}

std::string DominionState::ToString() const {
  std::string str;
  return str;
}

bool DominionState::GameFinished() const {
  SupplyPile provincePile = supply_piles_.find("province")->second;
  int num_empty_piles = absl::c_count_if(supply_piles_,[](std::pair<std::string,SupplyPile> pair){
    return pair.second.Empty();
  });
  return provincePile.Empty() || num_empty_piles == 3;
}

bool DominionState::IsTerminal() const {
  return is_terminal_;
}

void DominionState::DoApplyInitialSupplyChanceAction(Action action_id){
  const Card* card = GetCard(action_id);
  players_.at(current_player_).AddToDrawPile(card);
  if(EachPlayerReceivedInitSupply()){
    current_player_ = 0;
    absl::c_for_each(players_,[](PlayerState& player){
      player.DrawHand(kHandSize);
    });
  }else {
    auto player_itr = absl::c_find_if(players_,[](PlayerState player_state){
      return player_state.GetAllCards().size() < kInitSupply;
    });
    if(player_itr != players_.end()){
      current_player_ = player_itr->GetId();
    }
  }
}

void DominionState::DoApplyAddDiscardPileToDrawPile(Action action_id){
  auto player_itr = absl::c_find_if(players_,[](PlayerState player){
    return player.GetAddDiscardPileToDrawPile();
  });
  const Card* card = GetCard(action_id);
  player_itr->RemoveFromPile(card,DISCARD);
  player_itr->AddToDrawPile(card);
  bool discard_pile_empty = player_itr->GetDiscardPile().empty();
  player_itr->SetAddDiscardPileToDrawPile(!discard_pile_empty);
  if(discard_pile_empty){
      player_itr->DrawHand(player_itr->GetNumRequiredCards());
  }
}

void DominionState::DoApplyChanceAction(Action action_id){
  SPIEL_CHECK_EQ(CurrentPlayer(),kChancePlayerId);
  if(!EachPlayerReceivedInitSupply()){
      DoApplyInitialSupplyChanceAction(action_id);
  }else if(AddDiscardPileToDrawPile()){
    DoApplyAddDiscardPileToDrawPile(action_id);
  }
}

bool DominionState::AddDiscardPileToDrawPile() const {
  return absl::c_find_if(players_,[](PlayerState playerState){
    return playerState.GetAddDiscardPileToDrawPile();
  }) != players_.end();
}

bool DominionState::EachPlayerReceivedInitSupply() const {
  return absl::c_count_if(players_,[](PlayerState playerState){
    return playerState.GetAllCards().size() >= kInitSupply;
  }) == kNumPlayers;
}

Player DominionState::CurrentPlayer() const {
  if(!EachPlayerReceivedInitSupply()){
    return kChancePlayerId;
  }else if(AddDiscardPileToDrawPile()){
    return kChancePlayerId;
  }
  return current_player_;
}

std::vector<double> DominionState::Returns() const {
  return {0.0, 0.0};
}

std::string DominionState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void DominionState::ObservationTensor(Player player,
                                       absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
}

void DominionState::UndoAction(Player player, Action move) {
}

std::unique_ptr<State> DominionState::Clone() const {
  return std::unique_ptr<State>(new DominionState(*this));
}

DominionGame::DominionGame(const GameParameters& params)
    : Game(kGameType, params), kingdom_cards_(ParameterValue<std::string>("kingdom_cards")) {}

}  // namespace dominion
}  // namespace open_spiel
