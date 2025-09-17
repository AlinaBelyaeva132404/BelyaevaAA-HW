from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    MessageReactionHandler,
)

ALLOWED_NAMES = {
    "Юлия Найбауэр": ["Юлия Найбауэр", "Юля", "Юлия Фуррибауэр", "Юля Фуррибауэр", "Фуррибауэр"],
    "Ксения Кириллова": ["Ксения Кириллова", "Лума", "Ксюша Кириллова", "Ксюша", "Лума Кириллова"],
    "Александр Маслихов": ["Александр Маслихов", "Саша Маслихов", "Сашка", "Саша", "Сашка Маслихов"],
    "Алина Беляева": ["Алина Беляева", "крутая гейка", "Алина гейка", "Алиночка", "Алиночка Беляева", "Алинка"],
    "Михаил Стебло": ["Михаил Стебло", "Миша", "Миша Стебло"],
    "Мария Ефремова": ["Мария Ефремова", "Маша", "Маша Ефремова"],
    "Елизавета Шестакова": ["Елизавета Шестакова", "Вета", "Ветчина", "Ветчиназес", "Вета Шестакова",
                            "Ветчина Шестакова", "Ветчиназес Шестакова"],
    "Александра Лубошникова": ["Александра Лубошникова", "Саша", "Сашка", "Саша Лубошникова", "Сашка Лобушникова"],
    "Юлия Борисова": ["Юлия Борисова", "Лия", "Лия Борисова", "lia weile", "Лия Вейль"],
    "Валерия Закутина": ["Валерия Закутина", "Лера", "Лера Закутина"],
    "Валерия Болтанюк": ["Валерия Болтанюк", "Лэра", "Валэрия", "Валэрия Болтанюк", "Лэра Болтанюк"],
    "Кристина Казанцева": ["Кристина Казанцева", "Крис", "Крис Казанцева"],
    "Анастасия Исакова": ["Анастасия Исакова", "Настя", "Настя Исакова"],
    "Алиса Бакасова": ["Алиса Бакасова", "генан", "генан Бакасова"],
    "Данил Чинчикеев": ["Данил Чинчикеев", "Данил", "Даня", "Даня Чинчикеев"],
    "Софья Белкина": ["Софья Белкина", "Соня", "Соня Белкина"],
}

NICKNAME_MAP = {nickname: official for official, nicknames in ALLOWED_NAMES.items() for nickname in nicknames}

RESTRICTIONS = {
#    "Михаил Стебло": lambda slot, sched: ["Михаил Стебло: запрещённый участник"],
#    "Мария Ефремова": lambda slot, sched: ["Мария Ефремова: запрещённый участник"],
    "Ксения Кириллова": lambda slot, sched: list(filter(None, [
        "Ксения Кириллова:группа" if slot["group"] == 1 else None,
        "Ксения Кириллова:не с ФиПЛом" if not all(
            name in sched and sched[name]["group"] == slot["group"]
            for name in ["Алина Беляева", "Александр Маслихов", "Юлия Найбауэр"]
        ) else None
    ])),
    "Александр Маслихов": lambda slot, sched: list(filter(None, [
        "Александр Маслихов:в группе 2" if slot["group"] == 2 else None,
        "Александр Маслихов:не с ФиПЛом" if not all(
            name in sched and sched[name]["group"] == slot["group"]
            for name in ["Алина Беляева", "Ксения Кириллова", "Юлия Найбауэр"]
        ) else None
    ])),
    "Алиса Бакасова": lambda slot, _: list(filter(None, [
        "Алиса Бакасова:в группе 1" if slot["group"] == 1 else None,
        "Алиса Бакасова:в группе 4" if slot["group"] == 4 else None
    ])),
    "Валерия Болтанюк": lambda slot, _: list(filter(None, [
        "Валерия Болтанюк:в группе 1" if slot["group"] == 1 else None,
        "Валерия Болтанюк:в группе 4" if slot["group"] == 4 else None
    ])),
    "Юлия Найбауэр": lambda slot, sched: list(filter(None, [
        "Юлия Найбауэр:не с ФиПЛом" if not all(
            name in sched and sched[name]["group"] == slot["group"]
            for name in ["Алина Беляева", "Александр Маслихов", "Ксения Кириллова"]
        ) else None
    ])),
    "Алина Беляева": lambda slot, sched: list(filter(None, [
        "Алина Беляева:не с ФиПЛом" if not all(
            name in sched and sched[name]["group"] == slot["group"]
            for name in ["Юлия Найбауэр", "Александр Маслихов", "Ксения Кириллова"]
        ) else None
    ])),
    "Елизавета Шестакова": lambda slot, sched: list(filter(None, [
        "Елизавета Шестакова:не в первой" if slot["group"] != 1 else None
    ])),
    "Александра Лубошникова": lambda slot, sched: list(filter(None, [
        "Александра Лубошникова:группа" if slot["group"] == 1 else None,
        "Александра Лубошникова:не с Соней" if not all(
            name in sched and sched[name]["group"] == slot["group"]
            for name in ["Софья Белкина"]
        ) else None
    ])),
    "Софья Белкина": lambda slot, sched: list(filter(None, [
        "Софья Белкина:не с Сашей" if not (
                "Александра Лубошникова" in sched and sched["Александра Лубошникова"]["group"] == slot["group"]
        ) else None
    ])),
    "Юлия Борисова": lambda slot, _: list(filter(None, [
        "Юлия Борисова:в третьей" if slot["group"] == 3 else None
    ])),
    "Валерия Закутина": lambda slot, sched: list(filter(None, [
        "Валерия Закутина:не с Даниилом" if not (
                "Данил Чинчикеев" in sched and sched["Данил Чинчикеев"]["group"] == slot["group"]
        ) else None
    ])),
    "Данил Чинчикеев": lambda slot, sched: list(filter(None, [
        "Данил Чинчикеев:не с Лерой" if not (
                "Валерия Закутина" in sched and sched["Валерия Закутина"]["group"] == slot["group"]
        ) else None
    ])),
    "Анастасия Исакова": lambda slot, sched: list(filter(None, [
        "Анастасия Исакова:не с Крис" if not (
                "Кристина Казанцева" in sched and sched["Кристина Казанцева"]["group"] == slot["group"]
        ) else None
    ])),
    "Кристина Казанцева": lambda slot, _: list(filter(None, [
        "Кристина Казанцева:в четвёртой" if slot["group"] == 4 else None
    ]))
}

RESPONSE_DETAILS = {
    "Ксения Кириллова:группа": "Ты что, тупой? Ксюшу Кириллову нельзя ставить на 8, она не проснётся!!!",
    "Ксения Кириллова:не с ФиПЛом": "Ты серьёзно поставил Ксюшу не с ФиПЛом? Она же с ними пришла!",
    "Александр Маслихов:в группе 2": "Зачем ты поставил Александра Маслихова во вторую пятёрку, он в больнице будет в это время!!!!!",
    "Александр Маслихов:не с ФиПЛом": "Ты больной? На каком основании Маслихов не с ФиПЛом? Он же их староста, должен присматривать за своими идиотами!",
    "Алиса Бакасова:в группе 1": "Алиса спать хочет так-то, ей рано нельзя, придурок",
    "Алиса Бакасова:в группе 4": "Ты издеваешься? Алиса к этому времени сдохнет, ей на работу так-то надо!",
    "Валерия Болтанюк:в группе 1": "Серьёзно, Лэру в первую? Она же от страха на месте помрёт!",
    "Валерия Болтанюк:в группе 4": "Лэру в четвёртую? А кто её тревожку потом разгребать будет?",
    "Юлия Найбауэр:не с ФиПЛом": "Ты совсем логических связей в голове не имеешь? Юля только с ФиПЛом, иначе смерть",
    "Алина Беляева:не с ФиПЛом": "Ты вообще на каком основании Алину не с ФиПЛом поставил? Она так-то за компанию с ними пришла",
    "Александра Лубошникова:в первой": "Эм. Так-то Саша проспит к восьми",
    "Александра Лубошникова:не с Соней": "Ты поехавший? Вообще-то Саша только с Соней будет писать",
    "Юлия Борисова:в третьей": 'ТЫ ЧТО, РАСИСТ???? ЗАБЫЛ, ЧТО ЛИЯ ИЗ ЯКУТИИ???? ОНА УМРЁТ ОТ ЖАРЫ, ЕСЛИ ПОСТАВИТЬ ЕЁ НЕ С УТРА ИЛИ НЕ ВЕЧЕРОМ',
    "Валерия Закутина:не с Даниилом": "Ну, вообще-то, у Леры есть парень",
    "Данил Чинчикеев:не с Лерой": "О, да ты гений просто! Тебе Даня большое спасибо скажет за то, что разлучил его с дамой сердца",
    "Анастасия Исакова:не с Крис": "Ты а-дэ-ква-тный?? Так-то Настя только с Крис сдаёт",
    "Кристина Казанцева:в четвёртой": "Ты вообще в курсе, что Крис занятая женщина? Её нельзя ставить последней, у неё дома дела",
    "Михаил Стебло": "Ты свосем идиот? А ниче тот факт, что Миша не допущен до экзамена вообще?",
    "Елизавета Шестакова:не в первой": "Даун? Вета староста, она должна быть в первой четвёрке, чтобы отдать зачётки преподу!",
    "Мария Ефремова": "Ты серьёзно?! Ты думал, что у Маши Ефремовой не будет автомата???? Её вообще не надо никуда ставить, гнусное подобие на человека",
    "Софья Белкина:не с Сашей": "Ты совсем дебил? Соню можно ставить только с Сашей, они друг без друга не могут. Какой или какая Саша - думай своей пустой головой сам, одногруппники не дали тебе подсказок",

}

RESPONSES = {

}

OFFICIAL_NAMES = list(ALLOWED_NAMES.keys())

REQUIRED_NAMES = [name for name in OFFICIAL_NAMES if name not in ["Михаил Стебло", "Мария Ефремова"]]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ну давай посмотрим, как ТЫ справишься с моей рутиной во время экзаменов и зачётов, раз такой умный, выродок\n"
        "Скинь расписание в формате:\nИмя и Фамилия - номер четвёрки\n"
        "И только попробуй ошибиться - мне же такого шанса никто не давал\n"
        "Ах да, подсказки. Вот, какие предпочтения написали твои дорогие одногруппники: " "\n"
        "Ой, не понимаешь, почему тут пусто? ПОТОМУ ЧТО ТЕБЯ ВСЕ ПРОИГНОРИРОВАЛИ"
    )


def parse_message(text):
    lines = text.strip().split('\n')
    schedule = {}
    nicknames_used = []
    valid_format = True

    for line in lines:
        if '-' not in line:
            valid_format = False
            break
        try:
            name_part, group_part = map(str.strip, line.split('-', 1))
            group = int(group_part.split()[0])
            official_name = NICKNAME_MAP.get(name_part)
            if official_name:
                schedule[official_name] = {"group": group, "raw": name_part}
                if name_part != official_name:
                    nicknames_used.append(name_part)
            else:
                nicknames_used.append(name_part)
        except ValueError:
            valid_format = False
            break

    return schedule, nicknames_used, valid_format


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    casual_responses = {
        "игнорщик": "Вообще-то, в отличие от тебя, у меня есть хоть какие-то дела. Работай и не отвлекай меня",
        "а когда пересдача?": "Ты что, настолько тупой, что не сможешь сдать с первого раза? Иди учись",
        "че делать надо": "То, что делали все время ваши старосты??? Вы в курсе, как сложно ваши предпочтения неадекватные учесть??",
        "пошел нахуй": "Скажи <<триста>>",
        "триста": "Найди работу"
    }
    if text.lower() in casual_responses:
        await update.message.reply_text(casual_responses[text.lower()])
        return

    schedule, nicknames_used, valid_format = parse_message(text)

    if not valid_format and not schedule:
        await update.message.reply_text("Ты хоть что-нибудь нормально написал? Я вообще ничего не понял.")
        return

    errors = []

    # Проверяем, что в расписании только официальные имена
    for official_name, slot in schedule.items():
        raw_name = slot["raw"]
        if raw_name != official_name:
            errors.append(
                f"ТЫ ЧТО, СОВСЕМ ОТЪЕХАВЩИЙ???? ЭТО ОФИЦИАЛЬНЫЙ ДОКУМЕНТ, ПРЕПОДАВАТЕЛЬ ЗНАЕТ, ПО-ТВОЕМУ, КТО ТАКОЙ {raw_name.upper()}???????? Используй официальное имя, даун"
            )

    # Проверяем, что Михаил Стебло и Мария Ефремова не добавлены в расписание.
    if "Михаил Стебло" in schedule:
        errors.append(RESPONSE_DETAILS["Михаил Стебло"])
    if "Мария Ефремова" in schedule:
        errors.append(RESPONSE_DETAILS["Мария Ефремова"])

    # Проверяем, что все обязательные участники есть
    for person in REQUIRED_NAMES:
        if person not in schedule:
            errors.append(f"А ГДЕ {person.upper()}??? Ты забыл кого-то? УРОД!!!")

    # Проверяем ограничения на каждого участника
    for official_name, slot in schedule.items():
        if official_name in RESTRICTIONS:
            violations = RESTRICTIONS[official_name](slot, schedule)
            for v in violations:
                errors.append(RESPONSE_DETAILS.get(v, v))

    if errors:
        await update.message.reply_text("\n".join(errors))
    else:
        await update.message.reply_text("Господи, ничего нормально сделать с первого раза не можешь")


async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ну и уродство")


async def react_handler(update: Update, context: ContextTypes) -> None:
    if update.message_reaction.new_reaction:
        await update.message_reaction.chat.send_message("У тебя что, слов не хватает, чтобы нормально выразить мнение? Реакциями бросается, как школьник")


app = ApplicationBuilder().token("7854053235:AAFzbJsOkqBFY3ZdYF2dWnoFFtUWT-0IoHs").build()
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
app.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker))
app.add_handler(MessageReactionHandler(react_handler))


if __name__ == "__main__":
    try:
        print("Бот запускается...")
        app.run_polling()
    except Exception as e:
        print(f"Ошибка при запуске: {e}")
