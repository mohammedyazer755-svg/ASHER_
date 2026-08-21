package com.asher.companion.ui

import android.net.Uri
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.asher.companion.AsherApplication
import com.asher.companion.R
import com.asher.companion.pairing.PairingInvitation
import com.asher.companion.security.AuthorizationContext
import com.asher.companion.security.CompanionCapability
import com.asher.companion.security.RequestSource

/** Minimal, intentionally boring UI used to exercise the secure boundary. */
class MainActivity : AppCompatActivity() {
    private lateinit var applicationState: AsherApplication
    private lateinit var status: TextView
    private lateinit var invitationInput: EditText
    private lateinit var codeInput: EditText

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        applicationState = application as AsherApplication
        setContentView(buildContent())
        intent?.data?.let { handlePairingUri(it) }
    }

    private fun buildContent(): ViewGroup {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 36, 32, 24)
        }
        val scroll = ScrollView(this)
        val content = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val title = TextView(this).apply {
            text = getString(R.string.pairing_title)
            textSize = 22f
        }
        content.addView(title, weightParams())
        content.addView(TextView(this).apply {
            text = getString(R.string.pairing_description)
            textSize = 15f
        }, weightParams())
        status = TextView(this).apply { textSize = 16f }
        content.addView(status, weightParams())

        invitationInput = EditText(this).apply {
            hint = "Paste asher://pair link or invitation JSON"
            minLines = 3
            maxLines = 6
        }
        content.addView(invitationInput, weightParams())
        codeInput = EditText(this).apply {
            hint = "Code shown on the PC"
            inputType = android.text.InputType.TYPE_CLASS_TEXT
        }
        content.addView(codeInput, weightParams())

        content.addView(Button(this).apply {
            text = getString(R.string.pair_button)
            setOnClickListener { startPairing() }
        }, weightParams())
        content.addView(Button(this).apply {
            text = getString(R.string.authenticate_button)
            setOnClickListener { requestProtectedAction() }
        }, weightParams())
        scroll.addView(content)
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        refreshStatus()
        return root
    }

    private fun startPairing() {
        val decision = applicationState.permissionPolicy.evaluate(
            CompanionCapability.PAIR,
            AuthorizationContext(
                paired = false,
                userConfirmed = true,
                source = RequestSource.LOCAL_UI,
            ),
        )
        if (!decision.allowed) {
            status.text = decision.reason
            return
        }
        try {
            val invitation = parseInvitation(invitationInput.text.toString())
            val request = applicationState.pairingManager.begin(invitation, codeInput.text.toString())
            // A transport adapter owned by the host sends this request. The UI
            // displays only the public request, never a private key or channel key.
            status.text = "Pairing request created. Send this to the PC:\n${request.toJson()}"
        } catch (error: Exception) {
            status.text = "Pairing was not started: ${error.message ?: "invalid invitation"}"
        }
    }

    private fun requestProtectedAction() {
        val paired = applicationState.pairingManager.currentPeer() != null
        val decision = applicationState.permissionPolicy.evaluate(
            CompanionCapability.REMOTE_COMMAND,
            AuthorizationContext(paired = paired, source = RequestSource.LOCAL_UI),
        )
        if (decision.allowed || !paired) {
            status.text = decision.reason
            return
        }
        applicationState.biometricGate.authenticate(
            this,
            title = "Authorize ASHER command",
            subtitle = "This proof is one-shot and expires with the request",
        ) { result ->
            runOnUiThread {
                val consumed = result.verified && applicationState.biometricGate.consume(result.grant)
                status.text = if (consumed) {
                    "Authenticated. Confirm the exact action before sending it."
                } else {
                    if (result.verified) "Authentication grant was already consumed" else result.reason
                }
            }
        }
    }

    private fun parseInvitation(raw: String): PairingInvitation {
        val value = raw.trim()
        if (value.startsWith("{") && value.endsWith("}")) return PairingInvitation.fromJson(value)
        val uri = Uri.parse(value)
        if (uri.scheme != "asher" || uri.host != "pair") throw IllegalArgumentException("Invalid ASHER pairing link")
        val payload = uri.getQueryParameter("payload") ?: uri.getQueryParameter("invitation")
            ?: throw IllegalArgumentException("Pairing link has no invitation payload")
        val decoded = runCatching {
            android.util.Base64.decode(payload, android.util.Base64.URL_SAFE or android.util.Base64.NO_WRAP)
                .toString(Charsets.UTF_8)
        }.getOrElse { payload }
        return PairingInvitation.fromJson(decoded)
    }

    private fun handlePairingUri(uri: Uri) {
        if (uri.scheme == "asher" && uri.host == "pair") {
            invitationInput.setText(uri.toString())
        }
    }

    private fun refreshStatus() {
        status.text = if (applicationState.pairingManager.currentPeer() == null) {
            getString(R.string.status_unpaired)
        } else {
            "Paired securely. Protected actions still require local confirmation and device authentication."
        }
    }

    private fun weightParams(): LinearLayout.LayoutParams =
        LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
            bottomMargin = 16
        }
}
