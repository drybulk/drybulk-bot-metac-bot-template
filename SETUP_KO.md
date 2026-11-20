# 예측 봇 설정 안내

이 문서는 Metaculus FutureEval 봇 대회에 참가하는 예측 봇을 처음 켜는 방법과, 봇이 API 크레딧을 어떻게 나누어 쓰는지를 설명합니다. 봇은 Metaculus가 제공하는 공식 템플릿(`main.py`)을 그대로 두고, 그 위에 파일 두 개(`ensemble_bot.py`, `spend_ledger.py`)를 추가한 구조입니다.

## 1. 봇이 하는 일

1. GitHub Actions가 20분마다 봇을 실행합니다. 봇은 시즌 대회와 MiniBench에서 아직 예측하지 않은 질문을 가져옵니다.
2. 질문마다 Claude와 GPT가 각각 웹 검색으로 조사 보고서를 한 편씩 작성합니다. AskNews 키가 있으면 뉴스 검색 결과도 함께 사용합니다.
3. 같은 조사 보고서를 읽고 여러 모델이 따로 예측합니다. 기본 구성은 Claude Opus 5.5가 3회, GPT-6.2 Astra가 1회, GPT-6.2 Sol이 2회입니다.
4. 여섯 개의 예측을 하나로 합쳐서 Metaculus에 제출합니다. 예/아니오 질문은 중앙값을 사용합니다. 제출할 때 조사 내용과 각 모델의 추론이 댓글로 함께 게시되는데, 이것은 대회 규정이 요구하는 사항입니다.
5. 질문 하나를 처리할 때마다 지출액을 `state/spend_ledger.json`에 기록하고, 실행이 끝나면 이 파일을 저장소에 커밋합니다.

## 2. 준희 님이 하실 일

API 키와 토큰은 대화창에 붙여 넣지 마시고, 아래 4번에 적힌 GitHub 화면에 직접 입력해 주십시오.

1. **Metaculus 봇 계정 만들기.** https://www.metaculus.com/futureeval/participate/ 에서 본인 계정으로 로그인한 뒤 봇 계정을 만들고, 설정 화면에 표시되는 토큰을 복사합니다. 봇 이름은 순위표에 공개되므로, 실명을 드러내고 싶지 않으시면 실명과 무관한 이름을 사용하시면 됩니다. 한 사람이 상금을 받을 수 있는 봇은 하나입니다.
2. **템플릿 저장소 포크하기.** https://github.com/Metaculus/metac-bot-template 에서 Fork를 누릅니다. 공개(public) 저장소로 두시기를 권합니다. 비공개 저장소는 GitHub Actions 무료 사용량이 월 2,000분인데, 20분 간격 실행은 이 한도를 넘을 것으로 예상합니다. 공개 저장소는 Actions 사용이 무료이고, 키는 Secrets에 들어가므로 공개되지 않습니다. 다만 봇의 코드와 지출 기록 파일은 누구나 볼 수 있습니다.
3. **Claude가 포크에 접근하도록 허용하기.** https://github.com/apps/claude/installations/select_target 에서 방금 만든 포크를 Claude GitHub 앱의 접근 대상에 추가한 뒤, 저장소 이름을 스레드에 알려 주십시오. 그러면 제가 봇 코드를 그 저장소에 올립니다.
4. **키 입력하기.** 포크의 Settings → Secrets and variables → Actions → New repository secret에서 다음 세 개를 등록합니다.
   - `METACULUS_TOKEN`: 1번에서 복사한 봇 토큰
   - `ANTHROPIC_API_KEY`: Claude 구독에 포함된 월 100달러 크레딧을 쓰는 API 키
   - `OPENAI_API_KEY`: ChatGPT 구독에 포함된 월 100달러 크레딧을 쓰는 API 키
5. **초과 과금 막기.** Anthropic Console과 OpenAI Platform의 결제 설정에서 자동 충전(auto reload)이 꺼져 있는지 확인해 주십시오. 봇의 지출 기록은 추정값이므로, 실제 돈이 청구되지 않게 막는 최종 장치는 공급자 쪽의 크레딧 한도입니다.
6. **Claude 웹 검색 켜기.** Anthropic Console의 조직 설정에 웹 검색(Web search) 허용 항목이 있으면 켜 주십시오. 꺼져 있어도 봇은 멈추지 않고 GPT 쪽 조사만으로 예측하지만, 조사 품질은 낮아집니다.
7. **Actions 켜기와 시험 실행.** 포크의 Actions 탭을 열어서 `I understand my workflows, go ahead and enable them` 버튼이 보이면 누릅니다. 이 버튼을 누르지 않으면 20분 간격의 예약 실행이 시작되지 않을 수 있습니다. 그다음 `Test Bot`을 수동으로 실행합니다. 이 실행은 연습용 대회의 질문 4개를 예측하며, 크레딧을 두 공급자 합계로 2~5달러 정도 쓸 것으로 추정합니다. 실행이 끝나면 알려 주십시오. 제가 로그를 읽고 문제를 고칩니다.
8. **참가 양식 작성하기(필수).** 템플릿의 README는 예측을 제출하기 전에 모든 참가자가 https://forms.gle/aQdYMq9Pisrf1v7d8 양식의 첫 구역을 작성해야 한다고 안내합니다. 필수 질문은 3개라고 적혀 있습니다. 같은 양식의 뒷부분에서 Metaculus가 주는 무료 모델 크레딧을 신청할 수 있는데, 이 부분은 선택 사항이고 이번 시즌부터는 선별해서 지급한다고 공지되어 있습니다.

크레딧이 매월 1일이 아닌 날에 초기화된다면 그 날짜도 알려 주십시오. 아래 4절의 `BOT_ANTHROPIC_RESET_DAY`, `BOT_OPENAI_RESET_DAY` 값으로 반영합니다.

선택 사항이 한 가지 있습니다. AskNews 뉴스 검색은 봇 이름, 실명, LinkedIn 주소를 AskNews에 이메일로 보내야 사용할 수 있습니다. 개인정보를 보내야 하므로 처음에는 생략하고, 성적을 본 뒤에 결정하셔도 됩니다.

## 3. 지출을 관리하는 방식

두 공급자의 크레딧은 각각 월 100달러이고 이월되지 않습니다. 봇은 공급자마다 한 달 예산을 85달러로 잡습니다. 15달러는 비용 추정의 오차와 다른 용도를 위한 여유분입니다.

봇은 한 달이 지나간 비율과 지금까지 쓴 금액을 비교해서, 질문마다 공급자별 단계를 정합니다.

| 단계 | 조건 | Claude 쪽 구성 | GPT 쪽 구성 |
|---|---|---|---|
| full | 지출이 예산 × (경과 비율 + 25%) 이하 | Opus 5.5 예측 3회, 웹 검색 조사 | Astra 1회와 Sol 2회, 웹 검색 조사 |
| reduced | 지출이 예산 × (경과 비율 + 45%) 이하 | Opus 5.5 1회와 Sonnet 5.5 1회, 조사 | Sol 2회, 조사 |
| minimal | 지출이 예산 미만 | Sonnet 5.5 1회, 조사 | Sol 1회, 조사 |
| off | 지출이 예산 이상 | 사용하지 않음 | 사용하지 않음 |

예를 들어 한 달의 절반이 지난 시점에는 예산의 75%(약 64달러)까지 full 단계를 유지합니다. 한 공급자가 off가 되면 다른 공급자만으로 예측하고, 둘 다 off가 되면 그 질문은 예측하지 않습니다. 대회 점수는 예측한 질문에서 얻은 점수의 합계이므로, 예측하지 않은 질문은 감점 없이 0점으로 계산됩니다.

질문 하나에 드는 비용은 full 단계에서 공급자당 0.4~0.5달러로 추정합니다. 이 값은 실제 키로 실행해 보기 전에 토큰 수를 가정해서 계산한 것이므로, 시험 실행 뒤에 기록 파일의 실측값으로 바꾸겠습니다.

## 4. 바꿀 수 있는 설정

포크의 Settings → Secrets and variables → Actions → Variables에 아래 이름으로 값을 등록하면 기본값을 대신합니다. 등록하지 않으면 기본값을 사용합니다.

| 이름 | 기본값 | 의미 |
|---|---|---|
| `BOT_ANTHROPIC_MONTHLY_BUDGET` | 85 | Claude 쪽 월 예산(달러) |
| `BOT_OPENAI_MONTHLY_BUDGET` | 85 | GPT 쪽 월 예산(달러) |
| `BOT_ANTHROPIC_RESET_DAY` | 23 | Claude 크레딧이 초기화되는 날(이 저장소의 워크플로에 23일로 지정) |
| `BOT_OPENAI_RESET_DAY` | 5 | GPT 크레딧이 초기화되는 날(이 저장소의 워크플로에 5일로 지정) |
| `BOT_ANTHROPIC_STRONG` | anthropic/claude-opus-5-5 | Claude 쪽 주력 예측 모델 |
| `BOT_ANTHROPIC_MID` | anthropic/claude-sonnet-5-5 | Claude 쪽 조사 모델 겸 절약 단계의 예측 모델 |
| `BOT_OPENAI_STRONG` | openai/gpt-6.2-astra | GPT 쪽 주력 예측 모델 |
| `BOT_OPENAI_MID` | openai/gpt-6.2-sol | GPT 쪽 조사 모델 겸 보조 예측 모델 |

같은 크레딧을 다른 작업에도 쓰신다면 월 예산 값을 그만큼 낮춰 주십시오.

## 5. 확인한 것과 아직 확인하지 못한 것

키 없이 확인한 내용은 다음과 같습니다. `tests/offline_test.py`가 두 공급자의 API를 흉내 내는 로컬 서버를 띄우고 봇을 처음부터 끝까지 실행합니다. Python 3.13과 워크플로가 사용하는 3.11에서 49개 검사가 모두 통과했습니다.

- 예/아니오, 객관식, 수치 질문에서 예측이 만들어지고 합쳐집니다.
- 각 모델에 보내는 요청의 형식이 맞습니다. Claude 요청에는 temperature를 넣지 않고 effort를 high로 지정합니다.
- 비용이 호출마다 한 번씩만, 해당 공급자 앞으로 기록됩니다. 템플릿이 쓰는 비용 집계 방식은 같은 호출을 두 번 세는 경우가 있어서 사용하지 않았습니다.
- 예산을 넘긴 공급자는 호출하지 않고, 한 공급자의 API가 장애여도 다른 공급자만으로 예측합니다.
- 기록 파일이 손상되어도 봇이 멈추지 않습니다.

실제 키가 있어야 확인할 수 있는 내용은 다음과 같습니다. 이 부분은 2절 7번의 시험 실행에서 확인합니다.

- 실제 Metaculus에서 질문을 가져오고 예측을 제출하는 과정
- 실제 웹 검색 응답의 처리와 검색 비용의 집계. 흉내 낸 응답으로는 통과했지만, 실제 응답의 형식이 다를 수 있습니다.
- 두 계정의 크레딧으로 위 모델들과 웹 검색을 사용할 수 있는지 여부
- 질문 하나당 실제 비용과 소요 시간
- GitHub Actions가 기록 파일을 저장소에 커밋하는 단계

## 6. 파일 구성

| 파일 | 내용 |
|---|---|
| `ensemble_bot.py` | 봇 본체. 조사, 여러 모델의 예측, 실행 진입점 |
| `spend_ledger.py` | 지출 기록과 단계 결정 |
| `tests/offline_test.py`, `tests/mock_llm_server.py` | 키 없이 실행하는 시험 |
| `.github/workflows/run_bot_on_tournament.yaml` | 20분마다 대회 질문을 예측하고 지출 기록을 커밋 |
| `.github/workflows/test_bot.yaml` | 수동으로 실행하는 시험(질문 4개) |
| `.github/workflows/run_bot_on_metaculus_cup.yaml` | 예약 실행을 껐습니다. 이 워크플로는 템플릿 봇을 실행해서 지출 기록을 거치지 않고, Metaculus Cup에는 봇 상금이 없기 때문입니다. |
| `main.py` | Metaculus 템플릿 원본. 수정하지 않았습니다. 예측 프롬프트는 이 파일의 것을 그대로 사용합니다. |
