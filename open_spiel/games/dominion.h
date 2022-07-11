// // Copyright 2019 DeepMind Technologies Limited
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

// #ifndef OPEN_SPIEL_GAMES_DOMINION_H_
// #define OPEN_SPIEL_GAMES_DOMINION_H_

// #include <array>
// #include <map>
// #include <memory>
// #include <string>
// #include <vector>

// #include "open_spiel/spiel.h"

// namespace open_spiel {
// namespace dominion {

// // Constants.
// inline constexpr int kNumPlayers = 2;
// inline constexpr int kNumCards = 33;

// class Card {
//   public:
//     Card(std::string name);
//     std::string getName(){ return name_; }
//     int getId(){return id_; }
//     int getPlay() {return play_; }
//     int getBuy() {return buy_; }
//     int getDiscard() {return discard_;}
//     int getTrash() {return trash_;}
//     int getGain() {return gain_;}
//   protected:
//     int id_;
//     std::string name_;
//     int play_;
//     int buy_;
//     int discard_;
//     int trash_;
//     int gain_;
// };

// class PlayerState {
//   public:
//     PlayerState(int id){};
//     int victory_points() const;
//     std::vector<Card> getAllCards() const;
//     void addToDrawPile();
//     void drawHand();
//     bool hasCardInHand(Card card) const;
//     bool hasTreasureCardInHand() const;
//     bool hasActionCardInHand() const;
//     void play_treasure_card(TreasureCard card);
//     void buy_card(Card card);
//     void play_action_card(Card card);
//     void endPhase();
//     void endTurn();
//   private:
//     void addHandInPlayCardsToDiscardPile();
//     int id_;
//     std::vector<Card> drawPile_;
//     std::vector<Card> hand_;
//     std::vector<Card> discardPile_;
//     std::vector<Card> trashPile_;
//     std::vector<Card> cardsInPlay_;    
//     int vp_;
//     int buys_;
//     int coins_;
// };

// // State of an in-play game.
// class DominionState : public State {
//  public:
//   DominionState(std::shared_ptr<const Game> game);

//   DominionState(const DominionState&) = default;
//   DominionState& operator=(const DominionState&) = default;
//   Player CurrentPlayer() const override;
//   std::string ActionToString(Player player, Action action_id) const override;
//   std::string ToString() const override;
//   bool IsTerminal() const override;
//   std::vector<double> Returns() const override;
//   std::string InformationStateString(Player player) const override;
//   std::string ObservationString(Player player) const override;
//   void ObservationTensor(Player player,
//                          absl::Span<float> values) const override;
//   std::unique_ptr<State> Clone() const override;
//   std::vector<Action> LegalActions() const override;

//  protected:
//   void DoApplyAction(Action move) override;

//  private:
//   std::map<std::string, SupplyPile> supply_piles_;
//   std::vector<PlayerState> players_;
//   Player current_player_ = 0;
// };

// // Game object.
// class DominionGame : public Game {
//   public:
//     explicit DominionGame(const GameParameters& params);
//     int NumDistinctActions() const override;
//     std::unique_ptr<State> NewInitialState() const override {
//       return std::unique_ptr<State>(new DominionState(shared_from_this()));
//     }
//     int NumPlayers() const override { return kNumPlayers; }
//     double MinUtility() const override { return -1; }
//     double UtilitySum() const override { return 0; }
//     double MaxUtility() const override { return 1; }
//     std::vector<int> ObservationTensorShape() const override;
//     int MaxGameLength() const override;
//     bool randomKingdomCards() const; 
//     std::string kingdomCards() const;
//   private:
//     const bool use_random_kingdom_cards_;
//     const std::string kingdom_cards_;
// };

// }  // namespace dominion
// }  // namespace open_spiel

// #endif  // OPEN_SPIEL_GAMES_DOMINION_H_
// Copyright 2019 DeepMind Technologies Limited
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

#ifndef OPEN_SPIEL_GAMES_DOMINION_H_
#define OPEN_SPIEL_GAMES_DOMINION_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Simple game of Noughts and Crosses:
// https://en.wikipedia.org/wiki/Tic-tac-toe
//
// Parameters: none

namespace open_spiel {
namespace dominion {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kInitSupply = 10;
inline constexpr int kInitCoppers = 7;
inline constexpr int kInitEstates = 3;

inline constexpr int kNumCards = 33;
inline constexpr int kNumRows = 3;
inline constexpr int kNumCols = 3;
inline constexpr int kNumCells = kNumRows * kNumCols;
inline constexpr int kCellStates = 1 + kNumPlayers;  // empty, 'x', and 'o'.

// https://math.stackexchange.com/questions/485752/Dominion-state-space-choose-calculation/485852
inline constexpr int kNumberStates = 5478;

class Card {
  public:
    Card(int id, std::string name);
    std::string getName() const { return name_; };
    int GetId() const {return id_; }
    int GetPlay() const  {return play_; }
    int GetBuy() const {return buy_; }
    int GetDiscard() const {return discard_;}
    int GetTrash() const{ return trash_;}
    int GetGain() const {return gain_;}
    
  protected:
    int id_;
    std::string name_;
    int play_;
    int buy_;
    int discard_;
    int trash_;
    int gain_;
};

class TreasureCard : public Card {
  public:
    TreasureCard(int id, std::string name, int cost, int coins) : coins_(coins), cost_(cost), Card(id,name) {}
  private:
    int coins_;
    int cost_;
};
class VictoryCard : public Card {
  public:
    VictoryCard(int id, std::string name, int cost, int victory_points) : 
    cost_(cost), victory_points_(victory_points), Card(id,name) {};
  private:
    int cost_;
    int victory_points_;
};
class ActionCard : public Card {
  public:
    ActionCard(int id, std::string name, int cost, int add_actions=0, int add_buys=0,int coins=0, int add_cards=0): 
    cost_(cost), add_actions_(add_actions), add_buys_(add_buys),add_cards_(add_cards), coins_(coins), Card(id,name) {};
  private:
    int cost_;
    int add_actions_;
    int add_buys_;
    int add_cards_;
    int coins_;
};

const TreasureCard COPPER(1,"Copper",0,1);
const TreasureCard SILVER(2,"Silver",3,2);
const TreasureCard GOLD(3,"Gold",6,2);

const VictoryCard CURSE(4,"Curse",6,-1);
const VictoryCard ESTATE(5,"Estate",2,1);
const VictoryCard DUCHY(6,"Duchy",5,3);
const VictoryCard PROVINCE(7,"Province",8,6);

const ActionCard VILLAGE(8,"Village",3,2,0,0,1);
const ActionCard LABORATORY(9,"Laboratory",5,1,0,0,2);
const ActionCard FESTIVAL(10,"Festival",5,2,1,2,0);
const ActionCard MARKET(11,"Market",5,1,1,1,1);
const ActionCard SMITHY(12,"Smithy",4,0,0,0,3);
const ActionCard MILITIA(13,"Militia",4,0,0,0,0);
const VictoryCard GARDENS(14,"Gardens",4,0);
const ActionCard CHAPEL(15,"Chapel",2,0,0,0,0);
const ActionCard WITCH(16,"Witch",5,0,0,0,2);
const ActionCard WORKSHOP(17,"Workshop",3,0,0,0,0);
const ActionCard BANDIT(18,"Bandit",5,0,0,0,0);
const ActionCard REMODEL(19,"Remodel",4,0,0,0,0);
const ActionCard THRONE_ROOM(20,"Throne Room",4,0,0,0,0);
const ActionCard MONEYLENDER(21,"Moneylender",4,0,0,0,0);
const ActionCard POACHER(22,"Poacher",4,1,0,1,1);
const ActionCard MERCHANT(23,"Merchant",3,1,0,0,1);
const ActionCard CELLAR(24,"Cellar",2,1,0,0,0);
const ActionCard MINE(25,"Mine",5,0,0,0,0);
const ActionCard VASSAL(26,"Vassal",3,0,0,2,0);
const ActionCard COUNCIL_ROOM(27,"Council Room",5,1,4,0,0);
const ActionCard ARTISAN(28,"Artisan",6,0,0,0,0);
const ActionCard BUREAUCRAT(29,"Bureaucrat",4);
const ActionCard SENTRY(30,"Sentry",5,1,0,0,1);
const ActionCard HARBINGER(31,"Harbinger",3,1,0,0,1);
const ActionCard LIBRARY(32,"Library",5,0,0,0,0);
const ActionCard MOAT(33,"Moat",2,0,0,0,2);


const std::vector<Card> all_cards = {COPPER,SILVER,GOLD,CURSE,ESTATE,DUCHY,PROVINCE,VILLAGE,LABORATORY,
FESTIVAL,SMITHY,MILITIA,GARDENS,CHAPEL,WITCH,WORKSHOP,BANDIT,REMODEL,THRONE_ROOM,MONEYLENDER,POACHER,MERCHANT,
CELLAR,MINE,VASSAL,COUNCIL_ROOM,ARTISAN,BUREAUCRAT,SENTRY,HARBINGER,LIBRARY,MOAT
};



class SupplyPile {
  public: 
    SupplyPile(Card card, int qty) : card_(card), qty_(qty) {}; 
    bool isEmpty()const {return qty_ == 0;}
    int getQty(){return qty_;}
    Card getCard() {return card_; }
    void decrementSupplyPile() {qty_ -= 1;}
  private:
    int qty_;
    Card card_;
};

class PlayerState {
  public:
    PlayerState(Player id) : id_(id) {};
    int victory_points() const;
    Player GetId() const {return id_;};
    std::vector<Card> GetAllCards() const;
    std::vector<Card> GetDrawPile() const {return drawPile_;};
    void AddToDrawPile(Card card);
    void DrawHand();
    bool HasCardInHand(Card card) const;
    bool HasTreasureCardInHand() const;
    bool HasActionCardInHand() const;
    void Play_treasure_card(TreasureCard card);
    void Buy_card(Card card);
    void Play_action_card(Card card);
    void EndPhase();
    void EndTurn();
  private:
    void AddHandInPlayCardsToDiscardPile();
    Player id_;
    std::vector<Card> drawPile_;
    std::vector<Card> hand_;
    std::vector<Card> discardPile_;
    std::vector<Card> trashPile_;
    std::vector<Card> cardsInPlay_;    
    int vp_ = 0;
    int actions = 1;
    int buys_ = 1;
    int coins_ = 0;
};

// State of an in-play game.
class DominionState : public State {
 public:
  DominionState(std::shared_ptr<const Game> game);

  DominionState(const DominionState&) = default;
  DominionState& operator=(const DominionState&) = default;
  Player CurrentPlayer() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  // std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;
  void DoApplyAction(Action move) override;
  std::map<std::string,SupplyPile> getSupplyPiles()const {return supply_piles_;}
  std::vector<PlayerState>  getPlayers() {return players_;}
 private:
  Player current_player_ = 0;         // Player zero goes first
  bool EachPlayerReceivedInitSupply() const;
  Card GetCard(Action action_id) const; 
  void DoApplyChanceAction(Action action_id);
  std::map<std::string,SupplyPile> supply_piles_;
  std::vector<PlayerState> players_ {PlayerState(0),PlayerState(1)};
  int num_moves_ = 0;
};

// Game object.
class DominionGame : public Game {
 public:
  explicit DominionGame(const GameParameters& params);
  int NumDistinctActions() const override { return 0; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new DominionState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override {
    return {0, 0, 0};
  }
  int MaxGameLength() const override { return kNumCells; }
};


}  // namespace dominion
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_DOMINION_H_
