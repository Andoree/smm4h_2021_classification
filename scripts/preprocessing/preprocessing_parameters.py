import re

VERBOSE2CLASS_ID = {
    "NoADE": 0,
    "ADE": 1
}
USERNAME_REGEX = re.compile(r'@\w+')
URL_REGEX = re.compile(r'(http|https):[/a-zA-Z0-9.]+')
REPLACE_EN_EMOJIS_MAP = {
    '💉': ' injection ',
    '💀': ' skeleton ',
    '💊': ' pill ',
    '🔬': ' microscope ',
    '💩': ' shit ',
    '💤': ' sleep ',
    '🚬': ' cigarette ',
    '😂': ' laughing ',
    '😆': ' laughing ',
    '😴': 'sleeping',
    '😾': ' angry ',
    '😤': ' angry ',
    '😡': ' angry ',
    '😪': ' sleep ',
    '😩': ' crying ',
    '😨': ' fear ',
    '😰': ' fear ',
}
REPLACE_RU_EMOJIS_MAP = {
    '💉': ' укол ',
    '💀': ' череп ',
    '💊': ' таблетка ',
    '🔬': ' микроскоп ',
    '💩': ' дерьмо. ',
    '💤': ' сон ',
    '🚬': ' сигарета ',
    '😂': ' смешно ',
    '😆': ' смешно ',
    '😴': 'сон',
    '😾': ' гнев ',
    '😤': ' гнев ',
    '😡': ' гнев ',
    '😪': ' плач ',
    '😩': ' плач ',
    '😨': ' страшно ',
    '😰': ' страшно ',

}
EMOJI_MAPS_MAP = {
    'ru': REPLACE_RU_EMOJIS_MAP,
    'en': REPLACE_EN_EMOJIS_MAP,
    'fr': {}
}
REPLACE_AMP_MAP = {
    'ru': '&',
    'en': 'and',
    'fr': 'et'
}
