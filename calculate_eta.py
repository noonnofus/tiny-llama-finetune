def estimate_tuned(steps, currentStep, speed):
    """tuning이 완료 될때까지의 예상 시간 계산기"""
    if not isinstance(steps, int) or not isinstance(currentStep, int) or not isinstance(speed,(int, float)):
        raise ValueError("모든 파라미터는 int 혹은 float 타입이여야 합니다.")
    if speed <= 0:
        raise ValueError("해당 tuning은 끝나지 않을 것입니다. 다시 시작하거나 에러를 고친 뒤에 다시 시도해주세요.")
    leftSteps =  steps - currentStep
    timeInSec = leftSteps / speed
    return "당신의 tuning은 {}분 뒤 완료 될 것입니다.".format(round(timeInSec / 60, 2))

try:
    steps = input("전체 step을 입력해주세요: ")
    currentSteps = input("현재 step을 입력해주세요: ")
    speed = input("속도를 입력해주세요: ")
    time = estimate_tuned(int(steps), int(currentSteps), float(speed))
    print(time)
except ValueError as error:
    print('에러 발생: ', error)