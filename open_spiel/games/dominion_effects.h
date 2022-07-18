#ifndef OPEN_SPIEL_GAMES_DOMINION_EFFECTS__H_
#define OPEN_SPIEL_GAMES_DOMINION_EFFECTS_H_

#include "open_spiel/games/dominion.h"
#include "open_spiel/spiel.h"


namespace open_spiel{
namespace dominion{
namespace effects{

using namespace dominion;

class Effect{
  public:
    Effect(int id, std::string prompt): id_(id), prompt_(prompt){};
    virtual void Run(std::shared_ptr<DominionState> state, PlayerState& PlayerState){};
    virtual std::vector<Action> LegalActions(std::shared_ptr<DominionState> state, PlayerState& PlayerState) const {};
    virtual void DoApplyAction(Action action, std::shared_ptr<DominionState> state, PlayerState& PlayerState){};
    virtual std::string ActionToString(Action action, std::shared_ptr<DominionState> state, PlayerState& PlayerState) const {
      const Card* card = GetCard(action);
      std::string str = card->ActionToString(action);
      absl::StrAppend(&str,card->getName());
      return str;
    };
    virtual std::string ToString() const {return prompt_;}
  private:
    int id_;
    std::string prompt_;
};

class CellarEffect : public Effect {
  /* Discard any number of cards, then draw that many */
  public:
    CellarEffect(int id, std::string prompt) : Effect(id,prompt) {}
    void Run(std::shared_ptr<DominionState> state, PlayerState& PlayerState) {};
    std::vector<Action> LegalActions(std::shared_ptr<DominionState> state, PlayerState& PlayerState) const {};
    void DoApplyAction(Action action, std::shared_ptr<DominionState> state, PlayerState& PlayerState) {};
  private:
    int num_cards_discarded_ = 0;
};


 class EffectRunner {
  public: 
    EffectRunner(){};
    bool Active() const {};
    Player CurrentPlayer() const {};
    const Effect* CurrentEffect() const {return effects_.at(CurrentPlayer());};
    void AddEffect(Effect* effect, Player player){
      effects_[player] = effect;
    }
    void RemoveEffect(Player player){effects_[player] = nullptr;};
    std::vector<Action> LegalActions(std::shared_ptr<DominionState> state) const {
      PlayerState& player = state->GetPlayerState(CurrentPlayer());
      return effects_[CurrentPlayer()]->LegalActions(state,player);
    };
    void DoApplyAction(Action action, std::shared_ptr<DominionState> state) {
      PlayerState& player = state->GetPlayerState(CurrentPlayer());
      effects_[CurrentPlayer()]->DoApplyAction(action,state,player);
    }
    std::string ActionToString(Action action, std::shared_ptr<DominionState> state) const {
      PlayerState& player = state->GetPlayerState(CurrentPlayer());
      return effects_[CurrentPlayer()]->ToString(); 
    };
  private:
    std::array<Effect*,kNumPlayers> effects_;

};

const CellarEffect CELLAR_EFFECT(11,"Discard any number of cards, then +1 Card per card discarded ");

}
}
}
#endif  // OPEN_SPIEL_GAMES_DOMINION_EFFECTS_H_
