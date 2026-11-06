/*
    VocalHarmonizer
    by Ethan
    
    readme:
	this is only a juce wrapper around the DSP code in HarmonizerDSP.h... used in lieu of signalsmith stretch
    this would also be the third-ish/second version of the plugin. the first few just could not properly handle pitch shifting without causing pops!!!
    this also meant the project could be rewritten with efficiency

    MY coding is present in the other parts of the plugin

*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

HarmonizeAudioProcessor::ParameterState::ParameterLayout HarmonizeAudioProcessor::createParameterLayout()
{
    ParameterState::ParameterLayout layout;
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { "dryLevel", 1 }, "Dry Level",
        juce::NormalisableRange<float> { 0.0f, 1.0f }, 0.35f));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { "harmonyLevel", 1 }, "Harmony Level",
        juce::NormalisableRange<float> { 0.0f, 1.0f }, 0.8f));
    layout.add (std::make_unique<juce::AudioParameterFloat> (
        juce::ParameterID { "stereoSpread", 1 }, "Harmony Stereo Spread",
        juce::NormalisableRange<float> { 0.0f, 1.0f }, 0.7f));
    return layout;
}

HarmonizeAudioProcessor::HarmonizeAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor (BusesProperties()
#if ! JucePlugin_IsMidiEffect
        .withInput ("Input", juce::AudioChannelSet::mono(), true)
        .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
#endif
    ),
#else
    :
#endif
    parameters (*this, nullptr, "PARAMETERS", createParameterLayout())
{
    dryLevelParameter = parameters.getRawParameterValue ("dryLevel");
    harmonyLevelParameter = parameters.getRawParameterValue ("harmonyLevel");
    stereoSpreadParameter = parameters.getRawParameterValue ("stereoSpread");
    jassert (dryLevelParameter != nullptr && harmonyLevelParameter != nullptr && stereoSpreadParameter != nullptr);
}

HarmonizeAudioProcessor::~HarmonizeAudioProcessor() = default;

// boilerplate for juce
const juce::String HarmonizeAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool HarmonizeAudioProcessor::acceptsMidi() const
{
#if JucePlugin_WantsMidiInput
    return true;
#else
    return false;
#endif
}

bool HarmonizeAudioProcessor::producesMidi() const
{
#if JucePlugin_ProducesMidiOutput
    return true;
#else
    return false;
#endif
}

bool HarmonizeAudioProcessor::isMidiEffect() const
{
#if JucePlugin_IsMidiEffect
    return true;
#else
    return false;
#endif
}

double HarmonizeAudioProcessor::getTailLengthSeconds() const
{
    return harmonize::HarmonizerCore::getTailSeconds();
}

int HarmonizeAudioProcessor::getNumPrograms()
{
    return 1; 
}

int HarmonizeAudioProcessor::getCurrentProgram()
{
    return 0;
}

void HarmonizeAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String HarmonizeAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void HarmonizeAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

void HarmonizeAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused (samplesPerBlock);

    core.setParameters (dryLevelParameter->load(), harmonyLevelParameter->load(), stereoSpreadParameter->load());
    core.prepare (sampleRate);
    setLatencySamples (core.getLatencySamples());

    bypassGain.reset (sampleRate, 0.01);
    bypassGain.setCurrentAndTargetValue (1.0f);
    resetPending = false;
}

void HarmonizeAudioProcessor::releaseResources()
{
}

void HarmonizeAudioProcessor::reset()
{
    resetPending = true;
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool HarmonizeAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
#if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
#else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    if (layouts.getMainInputChannelSet() != juce::AudioChannelSet::mono())
        return false;

    return true;
#endif
}
#endif

void HarmonizeAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
	processInternal(buffer, midiMessages, false); // to process or not to process, that is the question
}

void HarmonizeAudioProcessor::processBlockBypassed (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    processInternal (buffer, midiMessages, true);
}

void HarmonizeAudioProcessor::processInternal (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages, bool bypassed)
{
    juce::ScopedNoDenormals noDenormals;

    const auto numSamples = buffer.getNumSamples();
    const auto numChannels = buffer.getNumChannels();

    if (numSamples == 0 || numChannels == 0)
        return;

    if (resetPending.exchange (false))
        core.reset();

    core.setParameters (dryLevelParameter->load(), harmonyLevelParameter->load(), stereoSpreadParameter->load());

    const float* input = getTotalNumInputChannels() > 0 ? buffer.getReadPointer (0) : nullptr;
    float* left = buffer.getWritePointer (0);
    float* right = numChannels > 1 ? buffer.getWritePointer (1) : nullptr;

    int position = 0;

    for (const auto metadata : midiMessages)
    {
        const auto eventPosition = juce::jlimit (position, numSamples, metadata.samplePosition);

        if (eventPosition > position)
        {
            core.process (input != nullptr ? input + position : nullptr,
                          left + position,
                          right != nullptr ? right + position : nullptr,
                          eventPosition - position);
            position = eventPosition;
        }

        if (metadata.numBytes <= 3)
            handleMidiMessage (metadata.getMessage());
    }

    if (position < numSamples)
        core.process (input != nullptr ? input + position : nullptr,
                      left + position,
                      right != nullptr ? right + position : nullptr,
                      numSamples - position);

    for (int channel = 2; channel < numChannels; ++channel)
        buffer.clear (channel, 0, numSamples);

    bypassGain.setTargetValue (bypassed ? 0.0f : 1.0f);

    if (bypassGain.isSmoothing() || bypassGain.getTargetValue() < 1.0f)
        bypassGain.applyGain (buffer, numSamples);
}

void HarmonizeAudioProcessor::handleMidiMessage (const juce::MidiMessage& message)
{
    float maxV = 0.95f;
	float minV = 0.2f; 
    if (message.isNoteOn())
		core.noteOn(message.getNoteNumber(), (minV + (maxV - minV) * message.getFloatVelocity())); // less dynamic is a good thing
    else if (message.isNoteOff())
        core.noteOff (message.getNoteNumber());
	else if (message.isSustainPedalOn()) // i don't necessarily need sustain in the plugin, apparently it's a good idea
        core.setSustain (true);
    else if (message.isSustainPedalOff())
        core.setSustain (false);
    else if (message.isAllNotesOff() || message.isAllSoundOff())
        core.allNotesOff();
    else if (message.isResetAllControllers())
        core.setSustain (false);
}

// a lot of this is boilerplate
bool HarmonizeAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* HarmonizeAudioProcessor::createEditor()
{
    return new HarmonizeAudioProcessorEditor (*this);
}

void HarmonizeAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    const auto state = parameters.copyState();

    if (auto xml = state.createXml())
        copyXmlToBinary (*xml, destData);
}

void HarmonizeAudioProcessor::setStateInformation (const void* data, int sizeInBytes) 
{
    const auto xml = getXmlFromBinary (data, sizeInBytes);

    if (xml != nullptr && xml->hasTagName (parameters.state.getType()))
        parameters.replaceState (juce::ValueTree::fromXml (*xml));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new HarmonizeAudioProcessor();
}
