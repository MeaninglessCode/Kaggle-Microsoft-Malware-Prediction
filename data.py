from sklearn.model_selection import train_test_split
from math import isnan
from numpy import where
import pandas as pd

cols_to_fe = ['EngineVersion', 'AppVersion', 'AvSigVersion', 'Census_OSVersion', 'OsVer']

cols_to_ohe = [
    'RtpStateBitfield', 'DefaultBrowsersIdentifier',
    'AVProductStatesIdentifier', 'AVProductsInstalled', 'AVProductsEnabled',
    'CountryIdentifier', 'CityIdentifier', 'GeoNameIdentifier',
    'OrganizationIdentifier',
    'LocaleEnglishNameIdentifier', 'Processor', 'OsBuild', 'OsSuite',
    'SmartScreen', 'Census_MDC2FormFactor', 'Census_OEMNameIdentifier',
    'Census_ProcessorCoreCount', 'Census_ProcessorModelIdentifier',
    'Census_PrimaryDiskTotalCapacity', 'Census_PrimaryDiskTypeName',
    'Census_HasOpticalDiskDrive', 'Census_TotalPhysicalRAM',
    'Census_ChassisTypeName', 'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_PowerPlatformRoleName', 'Census_InternalBatteryType',
    'Census_InternalBatteryNumberOfCharges', 'Census_OSEdition',
    'Census_OSInstallLanguageIdentifier', 'Census_GenuineStateName',
    'Census_ActivationChannel', 'Census_FirmwareManufacturerIdentifier',
    'Census_IsTouchEnabled', 'Census_IsPenCapable', 'IeVerIdentifier',
    'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer', 'SMode',
    'Wdft_RegionIdentifier', 'Census_ProcessorClass', 'Census_IsFlightingInternal',
    'Census_ThresholdOptIn', 'Census_IsWIMBootEnabled', 'PuaMode',
    'Census_FirmwareVersionIdentifier', 'Census_ProcessorManufacturerIdentifier',
    'Census_OEMModelIdentifier', 'Census_SystemVolumeTotalCapacity', 'ProductName',
    'Platform', 'OsPlatformSubRelease', 'OsBuildLab', 'SkuEdition',
    'Census_OSArchitecture', 'Census_OSBranch', 'Census_DeviceFamily',
    'Census_OSSkuName', 'Census_OSBuildRevision', 'Census_OSInstallTypeName',
    'Census_OSUILocaleIdentifier', 'Census_OSWUAutoUpdateOptionsName',
    'Census_FlightRing', 'Census_OSBuildNumber'
]

dtype = {
    'MachineIdentifier': 'object',
    'ProductName': 'category',
    'EngineVersion': 'category',
    'AppVersion': 'category',
    'AvSigVersion': 'category',
    'Census_OSVersion': 'category',
    'RtpStateBitfield': 'category',
    'IsSxsPassiveMode': 'int8',
    'DefaultBrowsersIdentifier': 'category',
    'AVProductStatesIdentifier': 'category',
    'AVProductsInstalled': 'category',
    'AVProductsEnabled': 'category',
    'CountryIdentifier': 'category',
    'CityIdentifier': 'category',
    'GeoNameIdentifier': 'category',
    'LocaleEnglishNameIdentifier': 'category',
    'OrganizationIdentifier': 'category',
    'Processor': 'category',
    'OsBuild': 'category',
    'OsBuildLab': 'category',
    'OsSuite': 'category',
    'OsVer': 'category',
    'OsPlatformSubRelease': 'category',
    'SMode': 'category',
    'Platform': 'category',
    'SmartScreen': 'category',
    'Census_MDC2FormFactor': 'category',
    'Census_OEMNameIdentifier': 'category',
    'Census_ProcessorCoreCount': 'category',
    'Census_ProcessorModelIdentifier': 'category',
    'Census_PrimaryDiskTotalCapacity': 'category',
    'Census_PrimaryDiskTypeName': 'category',
    'Census_HasOpticalDiskDrive': 'category',
    'Census_TotalPhysicalRAM': 'category',
    'Census_ChassisTypeName': 'category',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches': 'category',
    'Census_InternalPrimaryDisplayResolutionHorizontal': 'category',
    'Census_InternalPrimaryDisplayResolutionVertical': 'category',
    'Census_PowerPlatformRoleName': 'category',
    'Census_InternalBatteryType': 'category',
    'Census_InternalBatteryNumberOfCharges': 'category',
    'Census_OSEdition': 'category',
    'Census_OSInstallLanguageIdentifier': 'category',
    'Census_GenuineStateName': 'category',
    'Census_ActivationChannel': 'category',
    'Census_FirmwareManufacturerIdentifier': 'category',
    'Census_IsTouchEnabled': 'int8',
    'Census_IsPenCapable': 'int8',
    'Census_ProcessorClass': 'category',
    'Census_IsFlightingInternal': 'category',
    'Census_ThresholdOptIn': 'category',
    'IeVerIdentifier': 'category',
    'Census_IsPortableOperatingSystem': 'int8',
    'Census_IsSecureBootEnabled': 'int8',
    'Census_IsWIMBootEnabled': 'category',
    'Census_IsAlwaysOnAlwaysConnectedCapable': 'category',
    'Census_FirmwareVersionIdentifier': 'category',
    'Census_ProcessorManufacturerIdentifier': 'category',
    'Wdft_IsGamer': 'category',
    'PuaMode': 'category',
    'Wdft_RegionIdentifier': 'category',
    'Census_SystemVolumeTotalCapacity': 'category',
    'Census_OEMModelIdentifier': 'category',
    'HasTpm': 'int8',
    'IsBeta': 'int8',
    'HasDetections': 'int8',
    'SkuEdition': 'category',
    'Census_DeviceFamily': 'category',
    'Census_OSArchitecture': 'category',
    'Census_OSBranch': 'category',
    'Census_OSBuildNumber': 'category',
    'Census_OSBuildRevision': 'category',
    'Census_OSSkuName': 'category',
    'Census_OSInstallTypeName': 'category',
    'Census_OSUILocaleIdentifier': 'category',
    'Census_OSWUAutoUpdateOptionsName': 'category',
    'Census_FlightRing': 'category'
}

processed_csv_fp = 'assets/preprocessed/data.csv'
raw_csv_fp = 'assets/raw/train.csv'

label: str = 'HasDetections'
test_size: float = 0.3


def is_nan(x):
    if isinstance(x, float):
        if isnan(x):
            return True

    return False


def frequency_encode(df, cols: [str] = cols_to_fe):
    for col in cols:
        d = df[col].value_counts(dropna=False)
        df[f'{col}_FE'] = df[col].map(d)/d.max()

        # print(f'Frequency encoded {col}')

    df.drop(columns=cols, inplace=True)


'''
Statistical One-Hot Encoding will disregard attributes with more categories than make sense to.
It detects this using a trick from statics in which you assume a random sample, and upon each value test the hypothesis:
    H0: Prob(p=1) == m
    HA: Prob(p=1) != m
where p is the observed target_col rate given value is present, and m is a value between 0 and 1.

Then Central Limit Theory tells us that:
    z == (p-m)/std_dev(p) == 2*(p-m)*(n//2)
where n is #occurrences of value

which is transformed below to determine whether or not to translate.
'''


def one_hot_encode(df, cols: [str] = cols_to_ohe, target_col: str = label,
                   filter: float = 0.005, z: float = 5, m: float = 0.5):
    for col in cols:
        value_counts = df[col].value_counts(dropna=False)

        for x, n in value_counts.items():
            if n < filter * len(df):
                break
            entriesWithValue = df[col].isna() if is_nan(x) else df[col] == x

            p = df[entriesWithValue][target_col].mean()

            if 2 * abs(p - m) > (z / n//2):
                df[f'{col}_BE_{x}'] = entriesWithValue.astype('int8')

        # print(f'OHEncoded {col} and created {len(value_counts)} flags')
    df.drop(columns=cols, inplace=True)


def get_data():
    try:
        df = pd.read_csv(processed_csv_fp, index_col=0)
    except FileNotFoundError:
        df = pd.read_csv(raw_csv_fp, index_col=0, nrows=2000000, dtype=dtype)
        frequency_encode(df)
        one_hot_encode(df)
        df.to_csv(processed_csv_fp, float_format='%.4f')

    return train_test_split(df.drop(columns=[label]), df[label], test_size=test_size)
