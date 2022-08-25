add_executable (big_money_bot big_money_bot.cc
                ${OPEN_SPIEL_OBJECTS})

add_executable (big_money_bot_test big_money_bot_test.cc
                ${OPEN_SPIEL_OBJECTS} $<TARGET_OBJECTS:tests>)
add_test(big_money_bot_test big_money_bot_test)
