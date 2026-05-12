from app.data_providers import test_train_dataset
from processing.fixes import fix_action_type_target_leak

dataset = test_train_dataset()
df_train = dataset['train']
print(df_train['ACTION_TYPE'].value_counts())
fix_action_type_target_leak(df_train)
print(df_train['ACTION_TYPE'].value_counts())
