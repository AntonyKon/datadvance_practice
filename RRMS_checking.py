def notificator(func):
    import telebot

    def wrapper(*args):
        bot = telebot.TeleBot('1213161131:AAGbWfQTDsmfHOoEzz_y2QpNEalvZLMmcdI')
        my_id = 429854045

        result = None

        try:
            result = func(*args)
            bot.send_message(my_id, f"Расчет завершен: Результат: {result}")
        except:
            bot.send_message(my_id, "Что-то случилось с функцией расчета. Проверь входные данные")
        finally:
            bot.stop_bot()

        return result

    return wrapper


# @notificator
def RRMS_calc(model, validating_dataset, sample_size, check_number):
    RRMS_mean = 0

    for i in range(check_number):
        rows = [row for index, row in validating_dataset.sample(n=sample_size).iterrows()]
        # predicted_values = [model.calc(row.drop(labels=['price']).to_numpy())[0] for row in rows]
        exact_values = [row.at['price'] for row in rows]
        RRMS_mean += model.validate([row.drop(labels=['price']).to_numpy() for row in rows], exact_values)['RRMS'][0]

    return RRMS_mean / check_number
