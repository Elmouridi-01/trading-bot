class TradingSystemError(Exception):
    """الخطأ الأساسي للنظام"""
    pass

class CollectorError(TradingSystemError):
    """خطأ في جمع البيانات"""
    pass

class StrategyError(TradingSystemError):
    """خطأ في الاستراتيجية"""
    pass

class RiskError(TradingSystemError):
    """خطأ في إدارة المخاطر"""
    pass

class ExecutionError(TradingSystemError):
    """خطأ في تنفيذ الأوامر"""
    pass

class InsufficientFundsError(ExecutionError):
    """رصيد غير كافٍ"""
    pass

class PositionLimitError(RiskError):
    """تجاوز الحد الأقصى للمراكز المفتوحة"""
    pass

class DrawdownLimitError(RiskError):
    """تجاوز الحد الأقصى للخسارة"""
    pass