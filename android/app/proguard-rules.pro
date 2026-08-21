# Protocol model names are referenced by explicit serializers/reflection in a
# few host integrations. Keep their constructors and enum values in release.
-keep class com.asher.companion.protocol.** { *; }
-keep class com.asher.companion.pairing.** { *; }
-keep class com.asher.companion.security.** { *; }
