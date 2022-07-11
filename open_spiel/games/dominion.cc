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
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace dominion {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"dominion",
    /*long_name=*/"Dominion",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/false,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new DominionGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

Card::Card(int id, std::string name)
{
  id_ = id;
  name_ = name;
  play_ =  id;
  buy_ = 2 * id + kNumCards;
  trash_ = 3 * id + kNumCards;
  discard_ = 4 * id + kNumCards;
  gain_ = 5 * id + kNumCards;
}

std::map<std::string,SupplyPile> createTreasurePiles(){
  std::map<std::string,SupplyPile>treasurePiles;
  treasurePiles = {
    {"Copper",SupplyPile(COPPER,46)},
    {"Silver",SupplyPile(SILVER,40)},
    {"Gold",SupplyPile(GOLD,30)},
  };
  return treasurePiles;
}

std::map<std::string,SupplyPile> createVictoryPiles(){
  std::map<std::string,SupplyPile>victoryPiles;
  victoryPiles = {
    {"Curse",SupplyPile(CURSE,10)},
    {"Estate",SupplyPile(ESTATE,8)},
    {"Duchy",SupplyPile(DUCHY,8)},
    {"Province",SupplyPile(PROVINCE,8)}
  };
  return victoryPiles;
}


void PlayerState::AddToDrawPile(Card card){
  drawPile_.push_back(card);
}

std::vector<Card> PlayerState::GetAllCards() const {
  std::vector<Card> all_player_cards;
  all_player_cards.reserve(drawPile_.size() + hand_.size() + discardPile_.size() + trashPile_.size() + cardsInPlay_.size());
  all_player_cards.insert(all_player_cards.end(),drawPile_.begin(),drawPile_.end());
  all_player_cards.insert(all_player_cards.end(),hand_.begin(),hand_.end());
  all_player_cards.insert(all_player_cards.end(),discardPile_.begin(),discardPile_.end());
  all_player_cards.insert(all_player_cards.end(),trashPile_.begin(),trashPile_.end());
  all_player_cards.insert(all_player_cards.end(),cardsInPlay_.begin(),cardsInPlay_.end());
  return all_player_cards;
}

void DominionState::DoApplyAction(Action action_id) {
  if(IsChanceNode()){
    std::cout << action_id;
    DoApplyChanceAction(action_id);
  }
}

 std::vector<std::pair<Action, double>> DominionState::ChanceOutcomes() const {
  SPIEL_CHECK_EQ(CurrentPlayer(),kChancePlayerId);
  if(!EachPlayerReceivedInitSupply()){
    const std::vector<Card> draw_pile = players_.at(current_player_).GetDrawPile();      
    const int num_coppers = std::count_if(draw_pile.begin(),draw_pile.end(),[](Card card){
      return card.GetId() == COPPER.GetId();
    });
    const int num_estates = std::count_if(draw_pile.begin(),draw_pile.end(),[](Card card){
      return  card.GetId() == ESTATE.GetId();
    });
    const double total = kInitSupply - (num_coppers + num_estates);
    const double copper_p = (kInitCoppers - num_coppers) / total;
    const double estate_p = (kInitEstates - num_estates) / total;
    return std::vector<std::pair<Action,double>>{
      std::pair<Action,double>{COPPER.GetId(),copper_p},
      std::pair<Action,double>{ESTATE.GetId(),estate_p},
    };
  }
 }

std::vector<Action> DominionState::LegalActions() const {
  if (IsTerminal()) return {};
  std::vector<Action> moves;
  return moves;
}

std::string DominionState::ActionToString(Player player,
                                           Action action_id) const {
  return "";
}

DominionState::DominionState(std::shared_ptr<const Game> game) : State(game) {
  std::map<std::string,SupplyPile> victory_piles = createVictoryPiles();
  std::map<std::string,SupplyPile> treasure_piles = createTreasurePiles();
  supply_piles_.insert(victory_piles.begin(),victory_piles.end());
  supply_piles_.insert(treasure_piles.begin(),treasure_piles.end());

}

std::string DominionState::ToString() const {
  std::string str;
  return str;
}

bool DominionState::IsTerminal() const {
SupplyPile provincePile = supply_piles_.find("province")->second;
const int num_empty_piles = std::count_if(supply_piles_.begin(),supply_piles_.end(),[](std::pair<std::string,SupplyPile> entry){
  return entry.second.isEmpty();
});
 return provincePile.isEmpty() || num_empty_piles == 3;
}

Card DominionState::GetCard(Action action_id) const {
  return all_cards.at(action_id % all_cards.size() - 1);
}

void DominionState::DoApplyChanceAction(Action action_id){
  SPIEL_CHECK_EQ(CurrentPlayer(),kChancePlayerId);
  if(!EachPlayerReceivedInitSupply()){
    Card card = GetCard(action_id);
    players_.at(current_player_).AddToDrawPile(card);
    if(EachPlayerReceivedInitSupply()){
      current_player_ = 0;
    }else {
      auto playerState = std::find_if(players_.begin(),players_.end(),[](PlayerState player_state){
        return player_state.GetAllCards().size() < kInitSupply;
      });
      if(playerState != players_.end()){
        current_player_ = playerState->GetId();
      }
    }
  }
}

bool DominionState::EachPlayerReceivedInitSupply() const {
  return std::count_if(players_.begin(),players_.end(),[](PlayerState playerState){
    return playerState.GetAllCards().size() >= kInitSupply;
  }) == kNumPlayers;
}

Player DominionState::CurrentPlayer() const {
  if(!EachPlayerReceivedInitSupply()){
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
    : Game(kGameType, params) {}

}  // namespace dominion
}  // namespace open_spiel
