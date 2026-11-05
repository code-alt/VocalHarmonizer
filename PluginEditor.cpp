/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
HarmonizeAudioProcessorEditor::HarmonizeAudioProcessorEditor (HarmonizeAudioProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    const auto configureSlider = [] (juce::Slider& slider, juce::Label& label, const juce::String& text)
    {
        slider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 72, 20);
        label.setText (text, juce::dontSendNotification);
        label.setJustificationType (juce::Justification::centred);
    };

    configureSlider (dryLevelSlider, dryLevelLabel, "Dry Level");
    configureSlider (harmonyLevelSlider, harmonyLevelLabel, "Harmony Level");
    configureSlider (stereoSpreadSlider, stereoSpreadLabel, "Stereo Spread");

    addAndMakeVisible (dryLevelSlider);
    addAndMakeVisible (harmonyLevelSlider);
    addAndMakeVisible (stereoSpreadSlider);
    addAndMakeVisible (dryLevelLabel);
    addAndMakeVisible (harmonyLevelLabel);
    addAndMakeVisible (stereoSpreadLabel);
    auto& parameters = audioProcessor.getParameters();
    dryLevelAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
        parameters, "dryLevel", dryLevelSlider);
    harmonyLevelAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
        parameters, "harmonyLevel", harmonyLevelSlider);
    stereoSpreadAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (
        parameters, "stereoSpread", stereoSpreadSlider);

    setSize (480, 240);
}

HarmonizeAudioProcessorEditor::~HarmonizeAudioProcessorEditor()
{
}

//==============================================================================
void HarmonizeAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour::fromRGB (28, 31, 38));
}

void HarmonizeAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced (16);
    bounds.removeFromTop (18);
    const auto columnWidth = bounds.getWidth() / 3;
    const auto placeControl = [&bounds] (juce::Slider& slider, juce::Label& label, int x, int width)
    {
        label.setBounds (x, bounds.getY(), width, 28);
        slider.setBounds (x, bounds.getY() + 28, width, bounds.getHeight() - 28);
    };

    placeControl (dryLevelSlider, dryLevelLabel, bounds.getX(), columnWidth);
    placeControl (harmonyLevelSlider, harmonyLevelLabel, bounds.getX() + columnWidth, columnWidth);
    placeControl (stereoSpreadSlider, stereoSpreadLabel, bounds.getX() + columnWidth * 2,
                  bounds.getWidth() - columnWidth * 2);
}
